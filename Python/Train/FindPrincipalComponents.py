import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import RepeatedStratifiedKFold

# =====================================================
# EXPERIMENT SETTINGS
# =====================================================

# Stand / Sit_To_x / ...
TASK = "Stand" 

# If there is a PCA version of the dataset, set to its number like 0.95 , otherwise 0
USE_PCA = True
PCA = 0.95

N_SPLITS = 5
REPEATS = 5
RANDOM_SEED = 42

RF_TREES = 500
KNN_NEIGHBORS = 5

MASTER_PATH = r"/Users/mohammad/University/Bachelor Project/Final/Data/" + TASK + "/Master Features/MASTER_Features_" + TASK + "_PCA" + str(int(PCA*100)) + ".xlsx"
RESULTS_FOLDER = "Results/" + TASK + "/FindPrincipalComponents_" + TASK + ".xlsx"

os.makedirs(RESULTS_FOLDER, exist_ok=True)

OUTPUT_EXCEL = os.path.join(
    RESULTS_FOLDER,
    f"PCA_Dimension_Search_{int(PCA*100)}.xlsx"
)

# =====================================================
# LOAD FEATURES
# =====================================================

def load_features(master_path):

    df = pd.read_excel(master_path)

    df = df.drop(
        columns=[
            c for c in [
                "source_file",
                "prefix",
                "idx"
            ]
            if c in df.columns
        ],
        errors="ignore"
    )

    y = df["label"].astype(int).to_numpy()

    X = df.drop(columns=["label"]).copy()

    for c in X.columns:
        X[c] = pd.to_numeric(
            X[c],
            errors="coerce"
        )

    X = X.replace(
        [np.inf, -np.inf],
        np.nan
    )

    X = X.fillna(
        X.median(numeric_only=True)
    )

    return X.to_numpy(dtype=float), y

# =====================================================
# MODELS
# =====================================================

def get_models(seed):
    
    if USE_PCA:
        knn_steps = [
            ("clf",    KNeighborsClassifier(n_neighbors=KNN_NEIGHBORS))
        ]
        svm_steps = [
            ("clf",    SVC(kernel="rbf"))
        ]
        rf_steps = [
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
# SPECIFICITY
# =====================================================

def multiclass_specificity(y_true, y_pred):

    cm = confusion_matrix(y_true, y_pred)

    n_classes = cm.shape[0]

    specs = []

    for i in range(n_classes):

        TP = cm[i, i]

        FP = cm[:, i].sum() - TP

        FN = cm[i, :].sum() - TP

        TN = cm.sum() - TP - FP - FN

        if (TN + FP) == 0:
            specs.append(0)
        else:
            specs.append(
                TN / (TN + FP)
            )

    return np.mean(specs)


# =====================================================
# MAIN EVALUATION
# =====================================================
def evaluate_pca_dimensions():

    X, y = load_features(MASTER_PATH)

    total_components = X.shape[1]

    print(f"Total PCA components = {total_components}")

    skf = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    results = []

    for n_components in range(total_components, 0, -1):

        print(f"Testing first {n_components} components")

        X_current = X[:, :n_components]

        fold_acc = []
        fold_prec = []
        fold_rec = []
        fold_spec = []
        fold_f1 = []
                

        for train_idx, test_idx in skf.split(X_current, y):

            X_train = X_current[train_idx]
            X_test = X_current[test_idx]

            y_train = y[train_idx]
            y_test = y[test_idx]

            model = SVC(kernel="rbf")

            model.fit(X_train, y_train)

            pred = model.predict(X_test)

            acc = accuracy_score(y_test, pred)

            prec = precision_score(
                y_test,
                pred,
                average="macro",
                zero_division=0
            )

            rec = recall_score(
                y_test,
                pred,
                average="macro",
                zero_division=0
            )

            spec = multiclass_specificity(
                y_test,
                pred
            )

            f1 = f1_score(
                y_test,
                pred,
                average="macro",
                zero_division=0
            )

            fold_acc.append(acc)
            fold_prec.append(prec)
            fold_rec.append(rec)
            fold_spec.append(spec)
            fold_f1.append(f1)

        results.append({

            "Components": n_components,

            "Accuracy": np.mean(fold_acc) * 100,
            "Precision": np.mean(fold_prec) * 100,
            "Recall": np.mean(fold_rec) * 100,
            "Specificity": np.mean(fold_spec) * 100,
            "F1": np.mean(fold_f1) * 100,

            "Accuracy Std": np.std(fold_acc) * 100,
            "Precision Std": np.std(fold_prec) * 100,
            "Recall Std": np.std(fold_rec) * 100,
            "Specificity Std": np.std(fold_spec) * 100,
            "F1 Std": np.std(fold_f1) * 100,
        })

    results_df = pd.DataFrame(results)

    # =====================================================
    # PLOT ACCURACY VS PCA COMPONENTS
    # =====================================================

    metrics = [
        "Accuracy",
        "Precision",
        "Recall",
        "Specificity",
        "F1"
    ]

    for metric in metrics:

        plt.figure(figsize=(9,6))

        plt.plot(
            results_df["Components"],
            results_df[metric],
            linewidth=2.5,
            marker="o",
            markersize=5
        )

        plt.gca().invert_xaxis()

        plt.xlabel("Number of Principal Components", fontsize=12)
        plt.ylabel(metric + " (%)", fontsize=12)

        plt.title(
            f"SVM Performance vs Number of PCA Components ({metric})",
            fontsize=14,
            weight="bold"
        )

        plt.grid(True, linestyle="--", alpha=0.4)

        plt.xlim(
            results_df["Components"].max(),
            results_df["Components"].min()
        )

        plt.tight_layout()

        plt.savefig(
            os.path.join(
                RESULTS_FOLDER,
                metric.replace(" ", "_") + "_vs_PCA.png"
            ),
            dpi=600,
            bbox_inches="tight"
        )

        plt.close()
    
    results_df.to_excel(
        OUTPUT_EXCEL,
        index=False
    )

    print(results_df)

    return results_df


# =====================================================
# RUN
# =====================================================

if __name__ == "__main__":
    
    evaluate_pca_dimensions()