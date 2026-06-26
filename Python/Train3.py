import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    confusion_matrix
)

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier


# -----------------------------
# CONFIG
# -----------------------------
RANDOM_SEED = 42
K = 5  # fixed as you requested
PLOT_DIR = "plots_kfold5"
os.makedirs(PLOT_DIR, exist_ok=True)

SAVE_EXCEL = True
EXCEL_OUT = "kfold5_results.xlsx"


# -----------------------------
# Load features (MASTER or directory)
# -----------------------------
def load_features(master_path=None, features_dir=None):
    if master_path:
        df = pd.read_excel(master_path)
    else:
        if not features_dir:
            raise ValueError("Provide either master_path or features_dir.")
        files = sorted(glob.glob(os.path.join(features_dir, "Features_*.xlsx")))
        if not files:
            raise FileNotFoundError("No Features_*.xlsx found in the directory.")
        df = pd.concat([pd.read_excel(f) for f in files], ignore_index=True)

    if "label" not in df.columns:
        raise ValueError("label column not found.")

    # drop metadata if exists
    df = df.drop(columns=[c for c in ["source_file", "prefix", "idx"] if c in df.columns], errors="ignore")

    y = df["label"].astype(int).to_numpy()
    X = df.drop(columns=["label"]).copy()

    # numeric conversion
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")

    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True))

    return X.to_numpy(dtype=float), y, X.columns.tolist()


# -----------------------------
# Models (NO MLP)
# -----------------------------
def get_models(seed=42):
    return {
        "LogReg": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=5000))
        ]),
        "LinearSVM": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LinearSVC())
        ]),
        "SVM-RBF": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf"))
        ]),
        "KNN": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier(n_neighbors=5))
        ]),
        "RandomForest": RandomForestClassifier(n_estimators=600, random_state=seed),
        "ExtraTrees": ExtraTreesClassifier(n_estimators=900, random_state=seed),
        "GradientBoost": GradientBoostingClassifier(random_state=seed),
        "GaussianNB": GaussianNB(),
        "DecisionTree": DecisionTreeClassifier(random_state=seed),
    }


# -----------------------------
# K-Fold evaluation
# -----------------------------
def evaluate_kfold(X, y, k=5, seed=42):
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)

    rows = []
    cm_store = {}  # model -> list of confusion matrices

    for name, model in get_models(seed=seed).items():
        cm_store[name] = []
        for fold, (tr, te) in enumerate(skf.split(X, y), start=1):
            Xtr, ytr = X[tr], y[tr]
            Xte, yte = X[te], y[te]

            model.fit(Xtr, ytr)
            pred = model.predict(Xte)

            rows.append({
                "model": name,
                "fold": fold,
                "accuracy": accuracy_score(yte, pred),
                "balanced_acc": balanced_accuracy_score(yte, pred),
                "macro_f1": f1_score(yte, pred, average="macro")
            })

            cm_store[name].append(confusion_matrix(yte, pred, labels=[0, 1, 2]))

    df = pd.DataFrame(rows)

    summary = (
        df.groupby("model")[["accuracy", "balanced_acc", "macro_f1"]]
          .agg(["mean", "std"])
          .sort_values(("accuracy", "mean"), ascending=False)
    )

    return df, summary, cm_store


# -----------------------------
# Plotting (no seaborn)
# -----------------------------
def plot_box(df, metric, outpath):
    means = df.groupby("model")[metric].mean().sort_values(ascending=False)
    models = means.index.tolist()
    data = [df.loc[df["model"] == m, metric].values for m in models]

    plt.figure()
    plt.boxplot(data, labels=models, showmeans=True)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel(metric)
    plt.title(f"K-Fold (K=5) distribution of {metric}")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def plot_bar_mean_std(df, metric, outpath):
    g = df.groupby("model")[metric].agg(["mean", "std"]).sort_values("mean", ascending=False)

    x = np.arange(len(g.index))
    plt.figure()
    plt.bar(x, g["mean"].values, yerr=g["std"].values)
    plt.xticks(x, g.index, rotation=45, ha="right")
    plt.ylabel(metric)
    plt.title(f"K-Fold (K=5) Mean ± Std of {metric}")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def plot_confusion_heatmap(cm_mean, outpath, title):
    # simple heatmap with imshow (no seaborn)
    plt.figure()
    plt.imshow(cm_mean, interpolation="nearest")
    plt.title(title)
    plt.colorbar()
    ticks = np.arange(3)
    plt.xticks(ticks, ["0", "1", "2"])
    plt.yticks(ticks, ["0", "1", "2"])
    plt.xlabel("Predicted")
    plt.ylabel("True")

    # write values
    for i in range(cm_mean.shape[0]):
        for j in range(cm_mean.shape[1]):
            plt.text(j, i, f"{cm_mean[i, j]:.2f}", ha="center", va="center")

    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    MASTER_PATH = r"/Users/mohammad/University/Bachelor Project/Final/Data/Stand/Features/MASTER_Features_Stand.xlsx"
    FEATURES_DIR = None

    X, y, feat_names = load_features(master_path=MASTER_PATH, features_dir=FEATURES_DIR)

    df_folds, summary, cm_store = evaluate_kfold(X, y, k=K, seed=RANDOM_SEED)

    print("\nK-Fold (K=5) summary (mean±std):")
    print(summary)

    # Save Excel
    if SAVE_EXCEL:
        with pd.ExcelWriter(EXCEL_OUT) as w:
            df_folds.to_excel(w, sheet_name="fold_scores", index=False)
            summary.to_excel(w, sheet_name="summary")
        print(f"\nSaved: {EXCEL_OUT}")

    # Plots
    plot_box(df_folds, "accuracy", os.path.join(PLOT_DIR, "box_accuracy.png"))
    plot_box(df_folds, "balanced_acc", os.path.join(PLOT_DIR, "box_balanced_acc.png"))
    plot_box(df_folds, "macro_f1", os.path.join(PLOT_DIR, "box_macro_f1.png"))

    plot_bar_mean_std(df_folds, "accuracy", os.path.join(PLOT_DIR, "bar_accuracy_mean_std.png"))
    plot_bar_mean_std(df_folds, "balanced_acc", os.path.join(PLOT_DIR, "bar_balanced_acc_mean_std.png"))
    plot_bar_mean_std(df_folds, "macro_f1", os.path.join(PLOT_DIR, "bar_macro_f1_mean_std.png"))

    # Best model confusion matrix mean (normalized by fold size implicitly via mean)
    best_model = summary.index[0]
    cm_list = cm_store[best_model]
    cm_mean = np.mean(np.stack(cm_list, axis=0).astype(float), axis=0)

    plot_confusion_heatmap(
        cm_mean,
        os.path.join(PLOT_DIR, f"confusion_mean_{best_model}.png"),
        title=f"Mean Confusion Matrix over 5 folds - {best_model}"
    )

    print(f"\nPlots saved in: {PLOT_DIR}")
    print(f"Best model by mean accuracy: {best_model}")
