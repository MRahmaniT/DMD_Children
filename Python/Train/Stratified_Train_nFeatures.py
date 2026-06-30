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

# If there is a PCA version of the dataset, set to its number like 0.95 , otherwise 0
USE_PCA = True
PCA = 0.95

N_Components = 5

N_SPLITS = 5
REPEATS = 5
RANDOM_SEED = 42

RF_TREES = 500
KNN_NEIGHBORS = 5

MASTER_PATH = r"/Users/mohammad/University/Bachelor Project/Final/Data/Stand/Master Features/MASTER_Features_Stand_PCA" + str(int(PCA*100)) + ".xlsx"
RESULTS_FOLDER = r"/Users/mohammad/University/Bachelor Project/Python/Results/CompareNumberOfPrincipalComponents"

os.makedirs(RESULTS_FOLDER, exist_ok=True)

OUTPUT_EXCEL = os.path.join(
    RESULTS_FOLDER,
    f"Results_All_PCA_Components_{int(PCA*100)}.xlsx"
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
def run_stratified_Kfold():

    X_all, y = load_features(MASTER_PATH)

    models = get_models(RANDOM_SEED)

    all_runs = []
    component_summary = []

    # skf = StratifiedKFold(
    #     n_splits=5,
    #     shuffle=True,
    #     random_state=RANDOM_SEED
    # )
    
    skf = RepeatedStratifiedKFold(
        n_splits=N_SPLITS,
        n_repeats=REPEATS,
        random_state=RANDOM_SEED
    )
    for n_components in range(1, N_Components + 1):

        print(f"\n========== {n_components} Components ==========")

        X = X_all[:, :n_components]

        for model_name in models:

            accs = []
            precs = []
            recalls = []
            specs = []
            f1s = []

            print(f"\nRunning {model_name}")

            fold = 1

            for tr, te in skf.split(X, y):

                Xtr = X[tr]
                ytr = y[tr]

                Xte = X[te]
                yte = y[te]

                model = get_models(RANDOM_SEED)[model_name]

                model.fit(Xtr, ytr)

                pred = model.predict(Xte)

                acc = accuracy_score(yte, pred)

                prec = precision_score(
                    yte,
                    pred,
                    average="macro",
                    zero_division=0
                )

                rec = recall_score(
                    yte,
                    pred,
                    average="macro",
                    zero_division=0
                )

                spec = multiclass_specificity(
                    yte,
                    pred
                )

                f1 = f1_score(
                    yte,
                    pred,
                    average="macro",
                    zero_division=0
                )

                accs.append(acc)
                precs.append(prec)
                recalls.append(rec)
                specs.append(spec)
                f1s.append(f1)

                all_runs.append({
                    "Model": model_name,
                    "Fold": fold,
                    "Accuracy": acc * 100,
                    "Precision": prec * 100,
                    "Recall": rec * 100,
                    "Specificity": spec * 100,
                    "F1": f1 * 100
                })

                fold += 1

            component_summary.append({
                "Components": n_components,
                "Model": model_name,
                "Accuracy": np.mean(accs) * 100,
                "Precision": np.mean(precs) * 100,
                "Recall": np.mean(recalls) * 100,
                "Specificity": np.mean(specs) * 100,
                "F1": np.mean(f1s) * 100
            })

    summary_df = pd.DataFrame(component_summary)

    metrics = [
        "Accuracy",
        "Precision",
        "Recall",
        "Specificity",
        "F1"
    ]

    for metric in metrics:

        plt.figure(figsize=(9,6))

        for model in summary_df["Model"].unique():

            temp = summary_df[
                summary_df["Model"] == model
            ]

            plt.plot(
                temp["Components"],
                temp[metric],
                linewidth=2,
                marker="o",
                markersize=3,
                label=model
            )

        plt.xlabel("Number of PCA Components")

        plt.ylabel(metric + " (%)")

        plt.title(metric + " vs Number of PCA Components")

        plt.grid(True, linestyle="--", alpha=0.5)

        plt.legend()

        plt.tight_layout()

        plt.savefig(

            os.path.join(
                RESULTS_FOLDER,
                metric + "_vs_PCA_Components.png"
            ),

            dpi=600

        )

        plt.close()
    
    all_runs_df = pd.DataFrame(all_runs)

    with pd.ExcelWriter(
            OUTPUT_EXCEL,
            engine="openpyxl") as writer:

        summary_df.to_excel(
            writer,
            sheet_name="Results",
            index=False
        )

        all_runs_df.to_excel(
            writer,
            sheet_name="All_Folds",
            index=False
        )

    print("\n")
    print("=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    print(summary_df.round(3))

    print("\nSaved:")
    print(os.path.abspath(OUTPUT_EXCEL))

# =====================================================
# RUN
# =====================================================

if __name__ == "__main__":

    run_stratified_Kfold()
    