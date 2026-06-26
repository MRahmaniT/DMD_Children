import os
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, f1_score, confusion_matrix
)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

# =====================================================
# SETTINGS
# =====================================================

USE_PCA = True        # Toggle PCA on/off here
PCA_VARIANCE = 0.95

MASTER_PATH = r"/Users/mohammad/University/Bachelor Project/Final/Data/Stand/Master Features/MASTER_Features_Stand.xlsx"

if USE_PCA:
    OUTPUT_EXCEL = "Results/Results_Comparison_Fixed_PCA" + str(int(PCA_VARIANCE*100)) + ".xlsx"
else:
    OUTPUT_EXCEL = "Results/Results_Comparison.xlsx"
    
TRAIN_PER_CLASS = 20
TEST_PER_CLASS = 5
REPEATS = 100
RANDOM_SEED = 42
RF_TREES = 500
KNN_NEIGHBORS = 5

# =====================================================
# LOAD FEATURES
# =====================================================

def load_features(master_path):
    df = pd.read_excel(master_path)
    df = df.drop(
        columns=[c for c in ["source_file", "prefix", "idx"] if c in df.columns],
        errors="ignore"
    )
    y = df["label"].astype(int).to_numpy()
    X = df.drop(columns=["label"]).copy()
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True))
    return X.to_numpy(dtype=float), y

# =====================================================
# SPLIT
# =====================================================

def fixed_per_class_split(X, y, train_per_class, test_per_class, seed):
    rng = np.random.default_rng(seed)
    train_idx, test_idx = [], []
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        if len(idx) < train_per_class + test_per_class:
            raise ValueError(f"Class {c} has only {len(idx)} samples")
        idx = rng.permutation(idx)
        train_idx.extend(idx[:train_per_class])
        test_idx.extend(idx[train_per_class:train_per_class + test_per_class])
    return np.array(train_idx), np.array(test_idx)

# =====================================================
# SPECIFICITY
# =====================================================

def multiclass_specificity(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    specs = []
    for i in range(cm.shape[0]):
        TP = cm[i, i]
        FP = cm[:, i].sum() - TP
        FN = cm[i, :].sum() - TP
        TN = cm.sum() - TP - FP - FN
        specs.append(TN / (TN + FP) if (TN + FP) > 0 else 0)
    return np.mean(specs)

# =====================================================
# MODELS — PCA is now INSIDE the pipeline
# =====================================================

def get_models(seed, use_pca, pca_variance):

    if use_pca:
        # PCA is part of the pipeline — it will be fit on training data only, and applied to test data separately
        knn_steps = [
            ("scaler", StandardScaler()),
            ("pca",    PCA(n_components=pca_variance, random_state=seed)),
            ("clf",    KNeighborsClassifier(n_neighbors=KNN_NEIGHBORS))
        ]
        svm_steps = [
            ("scaler", StandardScaler()),
            ("pca",    PCA(n_components=pca_variance, random_state=seed)),
            ("clf",    SVC(kernel="rbf"))
        ]
        rf_steps = [
            ("scaler", StandardScaler()),
            ("pca",    PCA(n_components=pca_variance, random_state=seed)),
            ("clf",    RandomForestClassifier(n_estimators=RF_TREES, random_state=seed))
        ]
    else:
        knn_steps = [
            ("scaler", StandardScaler()),
            ("clf",    KNeighborsClassifier(n_neighbors=KNN_NEIGHBORS))
        ]
        svm_steps = [
            ("scaler", StandardScaler()),
            ("clf",    SVC(kernel="rbf"))
        ]
        rf_steps = [
            # ("scaler", StandardScaler()),
            ("clf",    RandomForestClassifier(n_estimators=RF_TREES, random_state=seed))
        ]

    return {
        "KNN":           Pipeline(knn_steps),
        "SVM":           Pipeline(svm_steps),
        "Random Forest": Pipeline(rf_steps)
    }

# =====================================================
# MAIN EVALUATION
# =====================================================

def run_repeated():
    X, y = load_features(MASTER_PATH)
    print(f"Dataset shape: {X.shape}")
    print(f"Classes: {np.unique(y)}")
    print(f"PCA: {'ON (' + str(int(PCA_VARIANCE*100)) + '% variance)' if USE_PCA else 'OFF'}")

    all_runs, summary = [], []

    for model_name in ["KNN", "SVM", "Random Forest"]:
        accs, precs, recalls, specs, f1s = [], [], [], [], []
        print(f"\nRunning {model_name}...")

        for r in range(REPEATS):
            seed = RANDOM_SEED + r

            tr, te = fixed_per_class_split(
                X, y, TRAIN_PER_CLASS, TEST_PER_CLASS, seed
            )

            Xtr, ytr = X[tr], y[tr]
            Xte, yte = X[te], y[te]

            # ✅ get_models is called here with train/test already separated
            # Pipeline.fit() on Xtr means scaler+PCA only see training data
            # Pipeline.predict() on Xte applies the already-fitted scaler+PCA
            model = get_models(seed, USE_PCA, PCA_VARIANCE)[model_name]
            model.fit(Xtr, ytr)
            pred = model.predict(Xte)

            accs.append(accuracy_score(yte, pred))
            precs.append(precision_score(yte, pred, average="macro", zero_division=0))
            recalls.append(recall_score(yte, pred, average="macro", zero_division=0))
            specs.append(multiclass_specificity(yte, pred))
            f1s.append(f1_score(yte, pred, average="macro", zero_division=0))

            all_runs.append({
                "Model": model_name, "Run": r + 1,
                "Accuracy":    accs[-1]  * 100,
                "Precision":   precs[-1] * 100,
                "Recall":      recalls[-1] * 100,
                "Specificity": specs[-1] * 100,
                "F1":          f1s[-1]   * 100
            })

        summary.append({
            "Model":       model_name,
            "Accuracy":    np.mean(accs)    * 100,
            "Precision":   np.mean(precs)   * 100,
            "Recall":      np.mean(recalls) * 100,
            "Specificity": np.mean(specs)   * 100,
            "F1":          np.mean(f1s)     * 100
        })

    summary_df = pd.DataFrame(summary).sort_values("F1", ascending=False)

    os.makedirs(os.path.dirname(OUTPUT_EXCEL), exist_ok=True)
    with pd.ExcelWriter(OUTPUT_EXCEL, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="Summary", index=False)
        pd.DataFrame(all_runs).to_excel(writer, sheet_name="All_Runs", index=False)

    print("\n" + "="*70)
    print(f"FINAL SUMMARY ({'with PCA' if USE_PCA else 'without PCA'})")
    print("="*70)
    print(summary_df.round(3).to_string(index=False))
    print(f"\nSaved: {os.path.abspath(OUTPUT_EXCEL)}")

if __name__ == "__main__":
    run_repeated()