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
from sklearn.model_selection import StratifiedKFold, GridSearchCV


# =====================================================
# EXPERIMENT SETTINGS
# =====================================================

# 1. Choose task : Stand / Sit_To_Stand / Jump / Rise_Head / Climb_Box_Right_Foot
TASK = "Stand"

# 2. Did you use action detector on your data or not
DETECTED = False

# 3. Model settings
MODEL_NAME = "SVM"

# Hyperparameter search space
# Linear SVM uses only C. RBF SVM uses both C and gamma.
SVM_PARAM_GRID = [
    {
        "clf__kernel": ["linear"],
        "clf__C": [0.01, 0.1, 1, 10, 100]
    },
    {
        "clf__kernel": ["rbf"],
        "clf__C": [0.01, 0.1, 1, 10, 100],
        "clf__gamma": ["scale", "auto", 0.0001, 0.001, 0.01, 0.1, 1]
    }
]

# Metric used by GridSearchCV to select the best hyperparameters.
# GridSearchCV selects hyperparameters using classification accuracy.
HYPERPARAMETER_SCORING = "accuracy"

# 4. Nested cross-validation settings
# Outer CV estimates generalization performance.
# Inner CV selects the optimum hyperparameters using training data only.
N_SPLITS = 5
INNER_N_SPLITS = 4
RANDOM_SEED = 42

# Use all CPU cores during GridSearchCV. Set to 1 if memory usage is high.
N_JOBS = -1

# 5. PCA component search settings
# If START_COMPONENTS = None, code starts from 1 component.
START_COMPONENTS = 1

# If END_COMPONENTS = None, code tests up to maximum possible component.
END_COMPONENTS = 50

# 6. Figure quality
FIG_DPI = 600

# 7. Required accuracy threshold in percent
# Example: 80 means 80%
SHOW_ACCURACY_THRESHOLD = True
ACCURACY_THRESHOLD = 80.0

# Highlight PCA component numbers whose accuracy reaches the threshold
HIGHLIGHT_ACCEPTABLE_ACCURACY = True

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
        + "/DetectedAction/FindHyperParameter_PipelinePCA_"
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
        + "/Normal/FindHyperParameter_PipelinePCA_"
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

    # The classifier starts with default values because GridSearchCV
    # replaces kernel, C, and gamma using SVM_PARAM_GRID.
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=n_components)),
        ("clf", SVC())
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

    # -----------------------------------------------------
    # SAFE PCA LIMIT FOR NESTED CROSS-VALIDATION
    # -----------------------------------------------------
    # PCA is fitted inside each INNER training fold, not on the full dataset.
    # Therefore n_components must not exceed the number of samples in the
    # smallest inner-training fold.
    outer_cv_for_limit = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_SEED
    )

    smallest_inner_training_size = X.shape[0]

    for outer_train_idx, _ in outer_cv_for_limit.split(X, y):

        X_outer_train = X[outer_train_idx]
        y_outer_train = y[outer_train_idx]

        inner_cv_for_limit = StratifiedKFold(
            n_splits=INNER_N_SPLITS,
            shuffle=True,
            random_state=RANDOM_SEED
        )

        for inner_train_idx, _ in inner_cv_for_limit.split(
            X_outer_train,
            y_outer_train
        ):
            smallest_inner_training_size = min(
                smallest_inner_training_size,
                len(inner_train_idx)
            )

    max_possible_components = min(
        smallest_inner_training_size,
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

    if start_components > end_components:
        raise ValueError(
            f"START_COMPONENTS={start_components} is greater than the "
            f"safe nested-CV PCA limit={end_components}."
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
    print(
        "Smallest inner-training fold size:",
        smallest_inner_training_size
    )
    print(
        "Maximum safe PCA components for nested CV:",
        max_possible_components
    )
    print("Search from:", start_components)
    print("Search to:", end_components)

    outer_cv = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_SEED
    )

    inner_cv = StratifiedKFold(
        n_splits=INNER_N_SPLITS,
        shuffle=True,
        random_state=RANDOM_SEED
    )

    results = []
    all_folds = []
    hyperparameter_results = []

    for n_components in range(start_components, end_components + 1):

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

        for train_idx, test_idx in outer_cv.split(X, y):

            X_train = X[train_idx]
            X_test = X[test_idx]

            y_train = y[train_idx]
            y_test = y[test_idx]

            pipeline = get_pipeline(
                n_components=n_components
            )

            # Nested CV:
            # GridSearchCV sees only the outer-training fold.
            # The untouched outer-test fold is used once for evaluation.
            grid_search = GridSearchCV(
                estimator=pipeline,
                param_grid=SVM_PARAM_GRID,
                scoring=HYPERPARAMETER_SCORING,
                cv=inner_cv,
                n_jobs=N_JOBS,
                refit=True,
                return_train_score=False,
                error_score="raise"
            )

            grid_search.fit(
                X_train,
                y_train
            )

            pred = grid_search.predict(
                X_test
            )

            best_params = grid_search.best_params_

            best_kernel = best_params["clf__kernel"]
            best_c = float(best_params["clf__C"])
            best_gamma = best_params.get("clf__gamma", "not_used")

            fold_metrics = calculate_metrics(
                y_true=y_test,
                y_pred=pred,
                labels=labels
            )

            hyperparameter_results.append({
                "Components": n_components,
                "Outer Fold": fold,
                "Best Kernel": best_kernel,
                "Best C": best_c,
                "Best Gamma": best_gamma,
                "Best Inner CV Score": grid_search.best_score_ * 100,
                "Scoring": HYPERPARAMETER_SCORING
            })

            print(
                f"  Fold {fold}: kernel={best_kernel}, "
                f"C={best_c}, gamma={best_gamma}, "
                f"inner {HYPERPARAMETER_SCORING}="
                f"{grid_search.best_score_ * 100:.2f}%"
            )

            for metric_name in fold_metrics_storage:

                fold_metrics_storage[metric_name].append(
                    fold_metrics[metric_name]
                )

            all_folds.append({
                "Components": n_components,
                "Fold": fold,
                "Best Kernel": best_kernel,
                "Best C": best_c,
                "Best Gamma": best_gamma,
                "Best Inner CV Score": grid_search.best_score_ * 100,
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

    hyperparameter_results_df = pd.DataFrame(
        hyperparameter_results
    )

    # Count how often each hyperparameter combination was selected
    hyperparameter_frequency_df = (
        hyperparameter_results_df
        .groupby(
            ["Components", "Best Kernel", "Best C", "Best Gamma"],
            dropna=False
        )
        .size()
        .reset_index(name="Selection Count")
        .sort_values(
            ["Components", "Selection Count"],
            ascending=[True, False]
        )
        .reset_index(drop=True)
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
    
    metric_linestyles = {
        "Accuracy": "-",
        "Precision": "--",
        "Recall": "-.",
        "Specificity": ":",
        "F1": (0, (5, 2)),
        "Balanced Accuracy": (0, (3, 1, 1, 1))
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

    fig, ax = plt.subplots(figsize=(13, 7.5))

    # Accuracy is deliberately plotted last so that it remains
    # visible when multiple metric curves overlap.
    plot_order = [
        "Precision",
        "Recall",
        "Specificity",
        "F1",
        "Balanced Accuracy",
        "Accuracy"
    ]

    for metric in plot_order:

        is_accuracy = metric == "Accuracy"

        ax.plot(
            results_df["Components"],
            results_df[metric],
            color=metric_colors[metric],

            # Accuracy is thicker and placed above other curves
            linewidth=3.2 if is_accuracy else 2.0,
            linestyle=metric_linestyles[metric],

            marker=metric_markers[metric],
            markersize=7 if is_accuracy else 5.5,

            # A filled marker makes Accuracy easier to identify
            markerfacecolor=(
                metric_colors[metric]
                if is_accuracy
                else "white"
            ),
            markeredgecolor=metric_colors[metric],
            markeredgewidth=1.4,

            # Higher zorder means the Accuracy line is drawn on top
            zorder=10 if is_accuracy else 3,

            # Slight transparency helps identify overlapping curves
            alpha=1.0 if is_accuracy else 0.82,

            label=metric
        )


    # =====================================================
    # ACCURACY THRESHOLD
    # =====================================================

    if SHOW_ACCURACY_THRESHOLD:

        if not 0 <= ACCURACY_THRESHOLD <= 100:
            raise ValueError(
                "ACCURACY_THRESHOLD must be between 0 and 100."
            )

        # Horizontal threshold line
        ax.axhline(
            y=ACCURACY_THRESHOLD,
            color="black",
            linestyle="--",
            linewidth=2.0,
            alpha=0.9,
            zorder=8,
            label=f"Required Accuracy ({ACCURACY_THRESHOLD:.1f}%)"
        )

        # Shade the acceptable region above the threshold
        ax.axhspan(
            ACCURACY_THRESHOLD,
            100,
            alpha=0.06,
            zorder=0
        )

        # Label placed near the right side of the threshold line
        ax.annotate(
            f"Required threshold = {ACCURACY_THRESHOLD:.1f}%",
            xy=(x_max, ACCURACY_THRESHOLD),
            xytext=(-8, 8),
            textcoords="offset points",
            ha="right",
            va="bottom",
            fontsize=9.5,
            fontweight="bold",
            bbox={
                "boxstyle": "round,pad=0.3",
                "facecolor": "white",
                "edgecolor": "black",
                "alpha": 0.85
            },
            zorder=12
        )


    # =====================================================
    # HIGHLIGHT ACCEPTABLE ACCURACY POINTS
    # =====================================================

    if SHOW_ACCURACY_THRESHOLD and HIGHLIGHT_ACCEPTABLE_ACCURACY:

        acceptable_mask = (
            results_df["Accuracy"] >= ACCURACY_THRESHOLD
        )

        acceptable_results = results_df.loc[
            acceptable_mask,
            ["Components", "Accuracy"]
        ].copy()

        if not acceptable_results.empty:

            # Large circles around acceptable Accuracy points
            ax.scatter(
                acceptable_results["Components"],
                acceptable_results["Accuracy"],
                s=115,
                facecolors="none",
                edgecolors=metric_colors["Accuracy"],
                linewidths=2.2,
                zorder=15,
                label="Accuracy Above Threshold"
            )

            # Small vertical guide lines make the corresponding
            # component numbers easier to locate.
            for _, row in acceptable_results.iterrows():

                component = int(row["Components"])
                accuracy_value = float(row["Accuracy"])

                ax.vlines(
                    x=component,
                    ymin=ACCURACY_THRESHOLD,
                    ymax=accuracy_value,
                    color=metric_colors["Accuracy"],
                    linestyle=":",
                    linewidth=1.0,
                    alpha=0.45,
                    zorder=2
                )

            acceptable_components = (
                acceptable_results["Components"]
                .astype(int)
                .to_list()
            )

            print("\n")
            print("=" * 70)
            print("ACCURACY THRESHOLD SUMMARY")
            print("=" * 70)

            print(
                f"Required accuracy threshold: "
                f"{ACCURACY_THRESHOLD:.2f}%"
            )

            print(
                "PCA components reaching the threshold:",
                acceptable_components
            )

            print(
                "Number of acceptable component settings:",
                len(acceptable_components)
            )

        else:

            print("\n")
            print("=" * 70)
            print("ACCURACY THRESHOLD SUMMARY")
            print("=" * 70)

            print(
                f"No PCA component reached the required "
                f"accuracy threshold of {ACCURACY_THRESHOLD:.2f}%."
            )


    # =====================================================
    # AXES AND APPEARANCE
    # =====================================================

    ax.set_xlabel(
        "Number of Principal Components",
        fontsize=12,
        fontweight="bold"
    )

    ax.set_ylabel(
        "Score (%)",
        fontsize=12,
        fontweight="bold"
    )

    ax.set_title(
        "SVM Performance vs Number of PCA Components",
        fontsize=14,
        fontweight="bold"
    )

    ax.set_xlim(
        x_min,
        x_max
    )

    ax.set_xticks(
        np.arange(
            x_min,
            x_max + 1,
            5
        )
    )

    # Use an appropriate lower limit while keeping the upper limit at 100.
    minimum_metric_value = float(
        results_df[metrics].min().min()
    )

    y_lower_limit = max(
        0,
        np.floor((minimum_metric_value - 5) / 5) * 5
    )

    if SHOW_ACCURACY_THRESHOLD:
        y_lower_limit = min(
            y_lower_limit,
            max(0, ACCURACY_THRESHOLD - 5)
        )

    ax.set_ylim(
        y_lower_limit,
        101
    )

    ax.tick_params(
        axis="both",
        labelsize=10
    )

    ax.grid(
        True,
        which="major",
        linestyle="--",
        linewidth=0.7,
        alpha=0.35
    )

    ax.legend(
        loc="best",
        fontsize=9,
        frameon=True,
        ncol=2
    )

    fig.tight_layout()

    fig.savefig(
        os.path.join(
            FIGURES_FOLDER,
            "All_Metrics_vs_PCA.png"
        ),
        dpi=FIG_DPI,
        bbox_inches="tight"
    )

    plt.close(fig)
    # =====================================================
    # SAVE EXCEL
    # =====================================================

    settings_df = pd.DataFrame({
        "Setting": [
            "Task",
            "Detected",
            "Model",
            "Hyperparameter Scoring",
            "SVM Parameter Grid",
            "Outer CV Splits",
            "Inner CV Splits",
            "N Jobs",
            "Random Seed",
            "Number of Samples",
            "Number of Raw Features",
            "Max Safe PCA Components (Nested CV)",
            "Start Components",
            "End Components",
            "Show Accuracy Threshold",
            "Accuracy Threshold (%)",
            "Highlight Acceptable Accuracy",
            "Master Path"
        ],
        "Value": [
            TASK,
            DETECTED,
            MODEL_NAME,
            HYPERPARAMETER_SCORING,
            str(SVM_PARAM_GRID),
            N_SPLITS,
            INNER_N_SPLITS,
            N_JOBS,
            RANDOM_SEED,
            X.shape[0],
            X.shape[1],
            max_possible_components,
            start_components,
            end_components,
            SHOW_ACCURACY_THRESHOLD,
            ACCURACY_THRESHOLD,
            HIGHLIGHT_ACCEPTABLE_ACCURACY,
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

        hyperparameter_results_df.to_excel(
            writer,
            sheet_name="Best_Hyperparameters",
            index=False
        )

        hyperparameter_frequency_df.to_excel(
            writer,
            sheet_name="Hyperparameter_Frequency",
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
    print("SELECTED HYPERPARAMETERS")
    print("=" * 70)
    print(hyperparameter_frequency_df.to_string(index=False))

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
