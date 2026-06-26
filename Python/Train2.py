import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

# -----------------------------
# Config
# -----------------------------
RANDOM_SEED = 42
REPEATS = 10          # <<< زیادش کن (مثلا 1000 هم میتونی)
TRAIN_PER_CLASS = 20
TEST_PER_CLASS = 5

SAVE_EXCEL = True
EXCEL_OUT = "ML_results.xlsx"

PLOT_DIR = "plots_ml"
os.makedirs(PLOT_DIR, exist_ok=True)

# -----------------------------
# Load data
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

    df = df.drop(columns=[c for c in ["source_file", "prefix", "idx"] if c in df.columns], errors="ignore")

    y = df["label"].astype(int).to_numpy()
    X = df.drop(columns=["label"]).copy()

    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")

    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True))

    return X.to_numpy(dtype=float), y, X.columns.tolist()

# -----------------------------
# Split: fixed count per class
# -----------------------------
def fixed_per_class_split(y, train_per_class=20, test_per_class=5, seed=42):
    rng = np.random.default_rng(seed)
    classes = np.unique(y)
    train_idx, test_idx = [], []

    for c in classes:
        idx_c = np.where(y == c)[0]
        need = train_per_class + test_per_class
        if idx_c.size < need:
            raise ValueError(f"Class {c} has {idx_c.size} samples; need {need}.")
        perm = rng.permutation(idx_c)
        train_idx.extend(perm[:train_per_class].tolist())
        test_idx.extend(perm[train_per_class:train_per_class+test_per_class].tolist())

    return np.array(train_idx, dtype=int), np.array(test_idx, dtype=int)

# -----------------------------
# Models (10)
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
        "RandomForest": RandomForestClassifier(n_estimators=500, random_state=seed),
        "ExtraTrees": ExtraTreesClassifier(n_estimators=800, random_state=seed),
        "GradientBoost": GradientBoostingClassifier(random_state=seed),
        "GaussianNB": GaussianNB(),
        "MLP": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=2500, random_state=seed))
        ]),
        "DecisionTree": DecisionTreeClassifier(random_state=seed),
    }

# -----------------------------
# Evaluate repeatedly
# -----------------------------
def evaluate_repeated(X, y, repeats=200, base_seed=42):
    model_names = list(get_models(seed=base_seed).keys())

    records = []
    for r in range(repeats):
        seed = base_seed + r
        tr, te = fixed_per_class_split(y, TRAIN_PER_CLASS, TEST_PER_CLASS, seed=seed)
        Xtr, ytr = X[tr], y[tr]
        Xte, yte = X[te], y[te]

        for name, model in get_models(seed=seed).items():
            model.fit(Xtr, ytr)
            pred = model.predict(Xte)

            records.append({
                "repeat": r,
                "seed": seed,
                "model": name,
                "accuracy": accuracy_score(yte, pred),
                "balanced_acc": balanced_accuracy_score(yte, pred),
                "macro_f1": f1_score(yte, pred, average="macro")
            })

    df = pd.DataFrame(records)
    return df

# -----------------------------
# Plotting helpers
# -----------------------------
def boxplot_metric(df_long, metric, outpath):
    models = sorted(df_long["model"].unique())
    data = [df_long.loc[df_long["model"] == m, metric].values for m in models]

    plt.figure()
    plt.boxplot(data, labels=models, showmeans=True)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel(metric)
    plt.title(f"Model comparison - {metric} (boxplot over repeats)")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()

def bar_mean_std(df_long, metric, outpath):
    g = df_long.groupby("model")[metric].agg(["mean", "std"]).sort_values("mean", ascending=False)

    plt.figure()
    x = np.arange(len(g.index))
    plt.bar(x, g["mean"].values, yerr=g["std"].values)
    plt.xticks(x, g.index, rotation=45, ha="right")
    plt.ylabel(metric)
    plt.title(f"Mean ± Std of {metric} over repeats")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()

def hist_best_model(df_long, metric, outpath):
    g = df_long.groupby("model")[metric].mean().sort_values(ascending=False)
    best = g.index[0]
    vals = df_long.loc[df_long["model"] == best, metric].values

    plt.figure()
    plt.hist(vals, bins=20)
    plt.xlabel(metric)
    plt.ylabel("count")
    plt.title(f"Distribution of {metric} - Best model: {best}")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()

# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    # یکی از این دو را تنظیم کن:
    MASTER_PATH = r"/Users/mohammad/University/Bachelor Project/Final/Data/Stand/Features/MASTER_Features_Stand.xlsx"
    FEATURES_DIR = None

    X, y, feat_names = load_features(master_path=MASTER_PATH, features_dir=FEATURES_DIR)

    df_long = evaluate_repeated(X, y, repeats=REPEATS, base_seed=RANDOM_SEED)

    # Summary table
    summary = (
        df_long.groupby("model")[["accuracy", "balanced_acc", "macro_f1"]]
        .agg(["mean", "std"])
        .sort_values(("accuracy", "mean"), ascending=False)
    )
    print("\nSummary (mean±std):")
    print(summary)

    # Save to excel
    if SAVE_EXCEL:
        with pd.ExcelWriter(EXCEL_OUT) as writer:
            df_long.to_excel(writer, sheet_name="all_runs", index=False)
            summary.to_excel(writer, sheet_name="summary")

        print(f"\nSaved excel: {EXCEL_OUT}")

    # Plots
    boxplot_metric(df_long, "accuracy", os.path.join(PLOT_DIR, "box_accuracy.png"))
    boxplot_metric(df_long, "balanced_acc", os.path.join(PLOT_DIR, "box_balanced_acc.png"))
    boxplot_metric(df_long, "macro_f1", os.path.join(PLOT_DIR, "box_macro_f1.png"))

    bar_mean_std(df_long, "accuracy", os.path.join(PLOT_DIR, "bar_accuracy_mean_std.png"))
    bar_mean_std(df_long, "balanced_acc", os.path.join(PLOT_DIR, "bar_balanced_acc_mean_std.png"))
    bar_mean_std(df_long, "macro_f1", os.path.join(PLOT_DIR, "bar_macro_f1_mean_std.png"))

    hist_best_model(df_long, "accuracy", os.path.join(PLOT_DIR, "hist_best_accuracy.png"))

    print(f"\nPlots saved in: {PLOT_DIR}")
