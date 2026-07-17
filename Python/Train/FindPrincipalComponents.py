import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    balanced_accuracy_score
)

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold


# =====================================================
# EXPERIMENT SETTINGS
# =====================================================

# 1. Choose task : Stand / Sit_To_Stand / Jump / Rise_Head / Climb_Box_Right_foot
TASK = "Rise_Head"

# 2. Did you use action detector on your data or not
DETECTED = True

# 3. Model settings
MODEL_NAME = "SVM"

SVM_KERNEL = "rbf"
SVM_C = 1
SVM_GAMMA = "scale"

# 4. Cross-validation settings
N_SPLITS = 5
RANDOM_SEED = 42

# 5. PCA component search settings
# If START_COMPONENTS = None, code starts from 1 component.
START_COMPONENTS = 1

# If END_COMPONENTS = None, code tests up to maximum possible component.
END_COMPONENTS = 50

# 6. Figure quality
FIG_DPI = 600


# =====================================================
# PATHS
# =====================================================

if DETECTED:

    MASTER_PATH = (
        r"/Users/mohammad/University/Bachelor Project/Final/DetectedActionData/"
        + TASK
        + "/Master Features/MASTER_Features_"
        + TASK
        + ".xlsx"
    )

    RESULTS_FOLDER = (
        "Results/"
        + TASK
        + "/DetectedAction/FindPrincipalComponents_PipelinePCA_"
        + TASK
    )

else:

    MASTER_PATH = (
        r"/Users/mohammad/University/Bachelor Project/Final/Data/"
        + TASK
        + "/Master Features/MASTER_Features_"
        + TASK
        + ".xlsx"
    )

    RESULTS_FOLDER = (
        "Results/"
        + TASK
        + "/Normal/FindPrincipalComponents_PipelinePCA_"
        + TASK
    )


FIGURES_FOLDER = os.path.join(
    RESULTS_FOLDER,
    "Figures"
)

os.makedirs(RESULTS_FOLDER, exist_ok=True)
os.makedirs(FIGURES_FOLDER, exist_ok=True)

OUTPUT_EXCEL = os.path.join(
    RESULTS_FOLDER,
    "PCA_Dimension_Search_PipelinePCA.xlsx"
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

    feature_names = X.columns.to_list()

    return X.to_numpy(dtype=float), y, feature_names


# =====================================================
# MODEL PIPELINE
# =====================================================

def get_pipeline(n_components):

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=n_components)),
        ("clf", SVC(
            kernel=SVM_KERNEL,
            C=SVM_C,
            gamma=SVM_GAMMA
        ))
    ])

    return pipeline


# =====================================================
# SPECIFICITY
# =====================================================

def multiclass_specificity(y_true, y_pred, labels):

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=labels
    )

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
# METRICS
# =====================================================

def calculate_metrics(y_true, y_pred, labels):

    acc = accuracy_score(
        y_true,
        y_pred
    )

    prec = precision_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    rec = recall_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    spec = multiclass_specificity(
        y_true,
        y_pred,
        labels=labels
    )

    f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    bal_acc = balanced_accuracy_score(
        y_true,
        y_pred
    )

    return {
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "Specificity": spec,
        "F1": f1,
        "Balanced Accuracy": bal_acc
    }


# =====================================================
# MAIN EVALUATION
# =====================================================

def evaluate_pca_dimensions():

    X, y, feature_names = load_features(
        MASTER_PATH
    )

    labels = np.unique(y)

    max_possible_components = min(
        X.shape[0] - 1,
        X.shape[1]
    )

    if END_COMPONENTS is None:

        end_components = max_possible_components

    else:

        end_components = min(
            END_COMPONENTS,
            max_possible_components
        )

    start_components = max(
        START_COMPONENTS,
        1
    )

    print("\n")
    print("=" * 70)
    print("DATASET INFORMATION")
    print("=" * 70)

    print("Task:", TASK)
    print("Detected:", DETECTED)
    print("Model:", MODEL_NAME)
    print("Master path:", MASTER_PATH)
    print("X shape:", X.shape)
    print("Number of samples:", X.shape[0])
    print("Number of raw features:", X.shape[1])
    print("Classes:", labels)
    print("Maximum possible PCA components:", max_possible_components)
    print("Search from:", start_components)
    print("Search to:", end_components)

    skf = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_SEED
    )

    results = []
    all_folds = []

    for n_components in range(start_components, end_components):

        print(f"\nTesting {n_components} PCA components inside pipeline")

        fold_metrics_storage = {
            "Accuracy": [],
            "Precision": [],
            "Recall": [],
            "Specificity": [],
            "F1": [],
            "Balanced Accuracy": []
        }

        fold = 1

        for train_idx, test_idx in skf.split(X, y):

            X_train = X[train_idx]
            X_test = X[test_idx]

            y_train = y[train_idx]
            y_test = y[test_idx]

            model = get_pipeline(
                n_components=n_components
            )

            model.fit(
                X_train,
                y_train
            )

            pred = model.predict(
                X_test
            )

            fold_metrics = calculate_metrics(
                y_true=y_test,
                y_pred=pred,
                labels=labels
            )

            for metric_name in fold_metrics_storage:

                fold_metrics_storage[metric_name].append(
                    fold_metrics[metric_name]
                )

            all_folds.append({
                "Components": n_components,
                "Fold": fold,
                "Accuracy": fold_metrics["Accuracy"] * 100,
                "Precision": fold_metrics["Precision"] * 100,
                "Recall": fold_metrics["Recall"] * 100,
                "Specificity": fold_metrics["Specificity"] * 100,
                "F1": fold_metrics["F1"] * 100,
                "Balanced Accuracy": fold_metrics["Balanced Accuracy"] * 100
            })

            fold += 1

        results.append({

            "Components": n_components,

            "Accuracy": np.mean(fold_metrics_storage["Accuracy"]) * 100,
            "Precision": np.mean(fold_metrics_storage["Precision"]) * 100,
            "Recall": np.mean(fold_metrics_storage["Recall"]) * 100,
            "Specificity": np.mean(fold_metrics_storage["Specificity"]) * 100,
            "F1": np.mean(fold_metrics_storage["F1"]) * 100,
            "Balanced Accuracy": np.mean(fold_metrics_storage["Balanced Accuracy"]) * 100,

            "Accuracy Std": np.std(fold_metrics_storage["Accuracy"]) * 100,
            "Precision Std": np.std(fold_metrics_storage["Precision"]) * 100,
            "Recall Std": np.std(fold_metrics_storage["Recall"]) * 100,
            "Specificity Std": np.std(fold_metrics_storage["Specificity"]) * 100,
            "F1 Std": np.std(fold_metrics_storage["F1"]) * 100,
            "Balanced Accuracy Std": np.std(fold_metrics_storage["Balanced Accuracy"]) * 100
        })

    results_df = pd.DataFrame(
        results
    )
    
    results_df = (
        results_df
        .sort_values(by="Components")
        .reset_index(drop=True)
    )

    all_folds_df = pd.DataFrame(
        all_folds
    )

    # =====================================================
    # FIND BEST COMPONENT NUMBER
    # =====================================================

    best_f1_row = results_df.loc[
        results_df["F1"].idxmax()
    ]

    best_accuracy_row = results_df.loc[
        results_df["Accuracy"].idxmax()
    ]

    best_balanced_accuracy_row = results_df.loc[
        results_df["Balanced Accuracy"].idxmax()
    ]

    best_f1_row_dict = best_f1_row.to_dict()
    best_accuracy_row_dict = best_accuracy_row.to_dict()
    best_balanced_accuracy_row_dict = best_balanced_accuracy_row.to_dict()

    best_summary_df = pd.DataFrame([
        {
            "Criterion": "Best F1",
            "Components": int(best_f1_row_dict["Components"]),
            "Score (%)": float(best_f1_row_dict["F1"])
        },
        {
            "Criterion": "Best Accuracy",
            "Components": int(best_accuracy_row_dict["Components"]),
            "Score (%)": float(best_accuracy_row_dict["Accuracy"])
        },
        {
            "Criterion": "Best Balanced Accuracy",
            "Components": int(best_balanced_accuracy_row_dict["Components"]),
            "Score (%)": float(best_balanced_accuracy_row_dict["Balanced Accuracy"])
        }
    ])

    print("\n")
    print("=" * 70)
    print("BEST COMPONENT SUMMARY")
    print("=" * 70)

    print(best_summary_df.round(3))

    # =====================================================
    # PLOTS
    # =====================================================

    metrics = [
        "Accuracy",
        "Precision",
        "Recall",
        "Specificity",
        "F1",
        "Balanced Accuracy"
    ]

    metric_colors = {
        "Accuracy": "#009E73",
        "Precision": "#E69F00",
        "Recall": "#0072B2",
        "Specificity": "#CC79A7",
        "F1": "#D55E00",
        "Balanced Accuracy": "#56B4E9"
    }

    metric_markers = {
        "Accuracy": "o",
        "Precision": "s",
        "Recall": "^",
        "Specificity": "D",
        "F1": "P",
        "Balanced Accuracy": "X"
    }

    x_min = int(results_df["Components"].min())
    x_max = int(results_df["Components"].max())

    for metric in metrics:

        plt.figure(figsize=(10, 6))

        plt.plot(
            results_df["Components"],
            results_df[metric],
            color=metric_colors[metric],
            linewidth=2.3,
            marker=metric_markers[metric],
            markersize=5.5,
            markerfacecolor="white",
            markeredgecolor=metric_colors[metric],
            markeredgewidth=1.3
        )

        plt.xlabel(
            "Number of Principal Components",
            fontsize=12,
            fontweight="bold"
        )

        plt.ylabel(
            metric + " (%)",
            fontsize=12,
            fontweight="bold"
        )

        plt.title(
            f"SVM Performance vs Number of PCA Components ({metric})",
            fontsize=14,
            fontweight="bold"
        )

        plt.xlim(x_min, x_max)

        plt.xticks(
            np.arange(x_min, x_max + 1, 5),
            fontsize=10
        )

        plt.yticks(fontsize=10)

        plt.grid(
            True,
            which="major",
            linestyle="--",
            linewidth=0.7,
            alpha=0.35
        )

        plt.tight_layout()

        plt.savefig(
            os.path.join(
                FIGURES_FOLDER,
                metric.replace(" ", "_") + "_vs_PCA.png"
            ),
            dpi=FIG_DPI,
            bbox_inches="tight"
        )

        plt.close()
        
    # =====================================================
    # ALL METRICS IN ONE PLOT
    # =====================================================

    plt.figure(figsize=(12, 7))

    for metric in metrics:

        plt.plot(
            results_df["Components"],
            results_df[metric],
            color=metric_colors[metric],
            linewidth=2,
            marker=metric_markers[metric],
            markersize=5,
            markerfacecolor="white",
            markeredgecolor=metric_colors[metric],
            markeredgewidth=1.2,
            label=metric
        )

    plt.xlabel(
        "Number of Principal Components",
        fontsize=12,
        fontweight="bold"
    )

    plt.ylabel(
        "Score (%)",
        fontsize=12,
        fontweight="bold"
    )

    plt.title(
        "SVM Performance vs Number of PCA Components",
        fontsize=14,
        fontweight="bold"
    )

    plt.xlim(x_min, x_max)

    plt.xticks(
        np.arange(x_min, x_max + 1, 5),
        fontsize=10
    )

    plt.yticks(fontsize=10)

    plt.grid(
        True,
        which="major",
        linestyle="--",
        linewidth=0.7,
        alpha=0.35
    )

    plt.legend(
        loc="best",
        fontsize=10,
        frameon=True,
        ncol=2
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            FIGURES_FOLDER,
            "All_Metrics_vs_PCA.png"
        ),
        dpi=FIG_DPI,
        bbox_inches="tight"
    )

    plt.close()

    # =====================================================
    # SAVE EXCEL
    # =====================================================

    settings_df = pd.DataFrame({
        "Setting": [
            "Task",
            "Detected",
            "Model",
            "SVM Kernel",
            "SVM C",
            "SVM Gamma",
            "N Splits",
            "Random Seed",
            "Number of Samples",
            "Number of Raw Features",
            "Max Possible PCA Components",
            "Start Components",
            "End Components",
            "Master Path"
        ],
        "Value": [
            TASK,
            DETECTED,
            MODEL_NAME,
            SVM_KERNEL,
            SVM_C,
            SVM_GAMMA,
            N_SPLITS,
            RANDOM_SEED,
            X.shape[0],
            X.shape[1],
            max_possible_components,
            start_components,
            end_components,
            MASTER_PATH
        ]
    })

    feature_names_df = pd.DataFrame({
        "Raw Feature Names": feature_names
    })

    with pd.ExcelWriter(
        OUTPUT_EXCEL,
        engine="openpyxl"
    ) as writer:

        settings_df.to_excel(
            writer,
            sheet_name="Settings",
            index=False
        )

        results_df.to_excel(
            writer,
            sheet_name="Results",
            index=False
        )

        all_folds_df.to_excel(
            writer,
            sheet_name="All_Folds",
            index=False
        )

        best_summary_df.to_excel(
            writer,
            sheet_name="Best_Summary",
            index=False
        )

        feature_names_df.to_excel(
            writer,
            sheet_name="Raw_Features",
            index=False
        )

    print("\n")
    print("=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)

    print(results_df.round(3))

    print("\nSaved Excel:")
    print(os.path.abspath(OUTPUT_EXCEL))

    print("\nSaved Figures:")
    print(os.path.abspath(FIGURES_FOLDER))

    return results_df


# =====================================================
# RUN
# =====================================================

if __name__ == "__main__":

    evaluate_pca_dimensions()