import os
import glob
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, classification_report, confusion_matrix

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

RANDOM_SEED = 42

# -----------------------------
# Load data
# Option A) load MASTER_Features_*.xlsx
# Option B) load all Features_*.xlsx (one-row) from a folder and concat
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
        df_list = [pd.read_excel(f) for f in files]
        df = pd.concat(df_list, ignore_index=True)

    if "label" not in df.columns:
        raise ValueError("label column not found. Make sure your feature files include a 'label' column.")

    # drop metadata if exists
    df = df.drop(columns=[c for c in ["source_file", "prefix", "idx"] if c in df.columns], errors="ignore")

    y = df["label"].astype(int).to_numpy()
    X = df.drop(columns=["label"]).copy()

    # numeric conversion
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")

    # handle inf/nan
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True))

    return X.to_numpy(dtype=float), y, X.columns.tolist()

# -----------------------------
# Split: Train 20/class, Test 5/class
# -----------------------------
def fixed_per_class_split(X, y, train_per_class=20, test_per_class=5, seed=42):
    rng = np.random.default_rng(seed)
    classes = np.unique(y)

    train_idx = []
    test_idx = []

    for c in classes:
        idx_c = np.where(y == c)[0]
        need = train_per_class + test_per_class
        if idx_c.size < need:
            raise ValueError(
                f"Class {c} has only {idx_c.size} samples, but need {need} (train {train_per_class} + test {test_per_class})."
            )

        perm = rng.permutation(idx_c)
        train_idx.extend(perm[:train_per_class].tolist())
        test_idx.extend(perm[train_per_class:train_per_class + test_per_class].tolist())

    return np.array(train_idx, dtype=int), np.array(test_idx, dtype=int)

# -----------------------------
# 10 ML models
# -----------------------------
def get_models(seed=42):
    return {
        "LogReg": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=5000))
        ]),
        "SVM-RBF": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf"))
        ]),
        "LinearSVM": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LinearSVC())
        ]),
        "RandomForest": RandomForestClassifier(n_estimators=400, random_state=seed),
        "DecisionTree": DecisionTreeClassifier(random_state=seed),
    }

# -----------------------------
# Run one split
# -----------------------------
def run_once(X, y, train_per_class=20, test_per_class=5, seed=42, verbose=True):
    tr, te = fixed_per_class_split(X, y, train_per_class, test_per_class, seed=seed)
    Xtr, ytr = X[tr], y[tr]
    Xte, yte = X[te], y[te]

    results = []
    for name, model in get_models(seed=seed).items():
        model.fit(Xtr, ytr)
        pred = model.predict(Xte)

        acc = accuracy_score(yte, pred)
        bacc = balanced_accuracy_score(yte, pred)
        f1m = f1_score(yte, pred, average="macro")

        results.append({
            "model": name,
            "accuracy_%": acc * 100,
            "balanced_acc_%": bacc * 100,
            "macro_f1": f1m
        })

        if verbose:
            print("=" * 70)
            print(f"Model: {name}")
            print(f"Train per class: {train_per_class} | Test per class: {test_per_class}")
            print(f"Accuracy       : {acc*100:.2f}%")
            print(f"Balanced Acc   : {bacc*100:.2f}%")
            print(f"Macro F1       : {f1m:.4f}")
            print("Confusion Matrix:")
            print(confusion_matrix(yte, pred))
            print("Report:")
            print(classification_report(yte, pred, digits=4))

    res_df = pd.DataFrame(results).sort_values("accuracy_%", ascending=False)
    return res_df

# -----------------------------
# Repeat (recommended)
# -----------------------------
def run_repeated(X, y, repeats=50, train_per_class=20, test_per_class=5, base_seed=42):
    model_names = list(get_models(seed=base_seed).keys())
    agg = {m: {"acc": [], "bacc": [], "f1": []} for m in model_names}

    for r in range(repeats):
        seed = base_seed + r
        tr, te = fixed_per_class_split(X, y, train_per_class, test_per_class, seed=seed)
        Xtr, ytr = X[tr], y[tr]
        Xte, yte = X[te], y[te]

        for name, model in get_models(seed=seed).items():
            model.fit(Xtr, ytr)
            pred = model.predict(Xte)
            agg[name]["acc"].append(accuracy_score(yte, pred))
            agg[name]["bacc"].append(balanced_accuracy_score(yte, pred))
            agg[name]["f1"].append(f1_score(yte, pred, average="macro"))

    rows = []
    for m in model_names:
        rows.append({
            "model": m,
            "acc_mean_%": np.mean(agg[m]["acc"]) * 100,
            "acc_std_%": np.std(agg[m]["acc"]) * 100,
            "bacc_mean_%": np.mean(agg[m]["bacc"]) * 100,
            "bacc_std_%": np.std(agg[m]["bacc"]) * 100,
            "f1_mean": np.mean(agg[m]["f1"]),
            "f1_std": np.std(agg[m]["f1"]),
        })

    return pd.DataFrame(rows).sort_values("acc_mean_%", ascending=False)

# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    # یکی از این دو تا را پر کن:
    MASTER_PATH = r"/Users/mohammad/University/Bachelor Project/Final/Data/Stand/Features/MASTER_Features_Stand.xlsx"
    FEATURES_DIR = None

    X, y, feat_names = load_features(master_path=MASTER_PATH, features_dir=FEATURES_DIR)

    print("\nSingle run (20 train + 5 test per class):")
    res = run_once(X, y, train_per_class=20, test_per_class=5, seed=RANDOM_SEED, verbose=True)
    print("\nSummary:")
    print(res.to_string(index=False))

    print("\nRepeated evaluation (mean±std over 50 random splits):")
    rep = run_repeated(X, y, repeats=50, train_per_class=20, test_per_class=5, base_seed=RANDOM_SEED)
    print(rep.to_string(index=False))
