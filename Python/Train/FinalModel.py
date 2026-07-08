import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    balanced_accuracy_score,
    classification_report,
    ConfusionMatrixDisplay
)

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import StratifiedKFold, cross_val_predict

# =====================================================
# EXPERIMENT SETTINGS
# =====================================================

# 1. Choose task : Stand / Sit_To_Stand / Jump / ...
TASK = "Stand"

# 2. Did you use action detector on your data or not
DETECTED = False

# 3. Choose final model
# Options: "SVM", "KNN", "Random Forest"
MODEL_NAME = "SVM"

# 4. Choose fixed number of PCA components inside pipeline
PCA_COMPONENTS = 3

# 5. Cross-validation settings
N_SPLITS = 5
RANDOM_SEED = 42

# 6. Model parameters
RF_TREES = 500
KNN_NEIGHBORS = 5

SVM_KERNEL = "rbf"
SVM_C = 1
SVM_GAMMA = "scale"

# 7. Save final model or not
SAVE_FINAL_MODEL = True

# 8. Figure quality
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
        r"/Users/mohammad/University/Bachelor Project/Results/"
        + TASK
        + "/Final/Final_Model_"
        + MODEL_NAME.replace(" ", "_")
        + "_PCA"
        + str(PCA_COMPONENTS)
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
        r"/Users/mohammad/University/Bachelor Project/Results/"
        + TASK
        + "/Final/Final_Model_"
        + MODEL_NAME.replace(" ", "_")
        + "_PCA"
        + str(PCA_COMPONENTS)
    )


FIGURES_FOLDER = os.path.join(
    RESULTS_FOLDER,
    "Figures"
)

MODEL_FOLDER = os.path.join(
    RESULTS_FOLDER,
    "Saved_Model"
)

os.makedirs(RESULTS_FOLDER, exist_ok=True)
os.makedirs(FIGURES_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)

OUTPUT_EXCEL = os.path.join(
    RESULTS_FOLDER,
    "Final_Model_Results.xlsx"
)

FINAL_MODEL_PATH = os.path.join(
    MODEL_FOLDER,
    "Final_Model_"
    + TASK
    + "_"
    + MODEL_NAME.replace(" ", "_")
    + "_PCA"
    + str(PCA_COMPONENTS)
    + ".pkl"
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
# MODEL
# =====================================================

def get_classifier(model_name, seed):

    if model_name == "SVM":

        clf = SVC(
            kernel=SVM_KERNEL,
            C=SVM_C,
            gamma=SVM_GAMMA,
            probability=True,
            random_state=seed
        )

    elif model_name == "KNN":

        clf = KNeighborsClassifier(
            n_neighbors=KNN_NEIGHBORS
        )

    elif model_name == "Random Forest":

        clf = RandomForestClassifier(
            n_estimators=RF_TREES,
            random_state=seed,
            class_weight="balanced"
        )

    else:

        raise ValueError(
            "Invalid MODEL_NAME. Choose from: 'SVM', 'KNN', 'Random Forest'"
        )

    return clf


def get_pipeline(model_name, seed, pca_components):

    clf = get_classifier(
        model_name=model_name,
        seed=seed
    )

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=pca_components)),
        ("clf", clf)
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

    prec_macro = precision_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    rec_macro = recall_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    f1_macro = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    prec_weighted = precision_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0
    )

    rec_weighted = recall_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0
    )

    f1_weighted = f1_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0
    )

    spec = multiclass_specificity(
        y_true,
        y_pred,
        labels=labels
    )

    bal_acc = balanced_accuracy_score(
        y_true,
        y_pred
    )

    metrics = {
        "Accuracy": acc * 100,
        "Precision Macro": prec_macro * 100,
        "Recall Macro": rec_macro * 100,
        "F1 Macro": f1_macro * 100,
        "Precision Weighted": prec_weighted * 100,
        "Recall Weighted": rec_weighted * 100,
        "F1 Weighted": f1_weighted * 100,
        "Specificity": spec * 100,
        "Balanced Accuracy": bal_acc * 100
    }

    return metrics


# =====================================================
# PLOTS
# =====================================================

def plot_main_metrics_bar(metrics_df):

    selected_metrics = [
        "Accuracy",
        "Precision Macro",
        "Recall Macro",
        "Specificity",
        "F1 Macro",
        "Balanced Accuracy"
    ]

    temp = metrics_df[
        metrics_df["Metric"].isin(selected_metrics)
    ].copy()

    plt.figure(figsize=(10, 6))

    plt.bar(
        temp["Metric"],
        temp["Score (%)"]
    )

    plt.ylim(0, 100)

    plt.ylabel("Score (%)")

    plt.title(
        "Final Model Performance - "
        + MODEL_NAME
        + " - PCA "
        + str(PCA_COMPONENTS)
    )

    for i, value in enumerate(temp["Score (%)"]):

        plt.text(
            i,
            value + 1,
            f"{value:.2f}%",
            ha="center",
            fontsize=10
        )

    plt.xticks(rotation=25)

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            FIGURES_FOLDER,
            "Final_Model_Main_Metrics_Bar.png"
        ),
        dpi=FIG_DPI
    )

    plt.close()


def plot_all_metrics_bar(metrics_df):

    plt.figure(figsize=(12, 6))

    plt.bar(
        metrics_df["Metric"],
        metrics_df["Score (%)"]
    )

    plt.ylim(0, 100)

    plt.ylabel("Score (%)")

    plt.title(
        "All Evaluation Metrics - "
        + MODEL_NAME
        + " - PCA "
        + str(PCA_COMPONENTS)
    )

    for i, value in enumerate(metrics_df["Score (%)"]):

        plt.text(
            i,
            value + 1,
            f"{value:.2f}%",
            ha="center",
            fontsize=9
        )

    plt.xticks(rotation=35, ha="right")

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            FIGURES_FOLDER,
            "Final_Model_All_Metrics_Bar.png"
        ),
        dpi=FIG_DPI
    )

    plt.close()


def plot_confusion_matrix(y_true, y_pred, labels):

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=labels
    )

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=labels
    )

    fig, ax = plt.subplots(figsize=(7, 6))

    disp.plot(
        ax=ax,
        cmap="Blues",
        values_format="d"
    )

    ax.set_title(
        "Confusion Matrix - "
        + MODEL_NAME
        + " - PCA "
        + str(PCA_COMPONENTS)
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            FIGURES_FOLDER,
            "Confusion_Matrix.png"
        ),
        dpi=FIG_DPI
    )

    plt.close()


def plot_normalized_confusion_matrix(y_true, y_pred, labels):

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=labels,
        normalize="true"
    )

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=labels
    )

    fig, ax = plt.subplots(figsize=(7, 6))

    disp.plot(
        ax=ax,
        cmap="Blues",
        values_format=".2f"
    )

    ax.set_title(
        "Normalized Confusion Matrix - "
        + MODEL_NAME
        + " - PCA "
        + str(PCA_COMPONENTS)
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            FIGURES_FOLDER,
            "Normalized_Confusion_Matrix.png"
        ),
        dpi=FIG_DPI
    )

    plt.close()


def plot_pca_explained_variance(final_pipeline):

    pca = final_pipeline.named_steps["pca"]

    explained_variance = pca.explained_variance_ratio_ * 100

    cumulative_variance = np.cumsum(
        explained_variance
    )

    pc_names = [
        "PC" + str(i + 1)
        for i in range(len(explained_variance))
    ]

    pca_df = pd.DataFrame({
        "Principal Component": pc_names,
        "Explained Variance (%)": explained_variance,
        "Cumulative Explained Variance (%)": cumulative_variance
    })

    plt.figure(figsize=(9, 6))

    plt.bar(
        pca_df["Principal Component"],
        pca_df["Explained Variance (%)"]
    )

    plt.plot(
        pca_df["Principal Component"],
        pca_df["Cumulative Explained Variance (%)"],
        marker="o",
        linewidth=2
    )

    plt.ylabel("Explained Variance (%)")

    plt.xlabel("Principal Components")

    plt.title(
        "PCA Explained Variance - "
        + str(PCA_COMPONENTS)
        + " Components"
    )

    plt.grid(
        True,
        linestyle="--",
        alpha=0.5
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            FIGURES_FOLDER,
            "PCA_Explained_Variance.png"
        ),
        dpi=FIG_DPI
    )

    plt.close()

    return pca_df


def plot_cv_fold_metrics(fold_results_df):

    metrics = [
        "Accuracy",
        "Precision Macro",
        "Recall Macro",
        "Specificity",
        "F1 Macro",
        "Balanced Accuracy"
    ]

    plt.figure(figsize=(11, 6))

    for metric in metrics:

        plt.plot(
            fold_results_df["Fold"],
            fold_results_df[metric],
            marker="o",
            linewidth=1.5,
            label=metric
        )

    plt.xlabel("Fold")

    plt.ylabel("Score (%)")

    plt.title(
        "Cross-Validation Fold Metrics - "
        + MODEL_NAME
        + " - PCA "
        + str(PCA_COMPONENTS)
    )

    plt.ylim(0, 100)

    plt.grid(
        True,
        linestyle="--",
        alpha=0.5
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            FIGURES_FOLDER,
            "CV_Fold_Metrics.png"
        ),
        dpi=FIG_DPI
    )

    plt.close()


# =====================================================
# MAIN
# =====================================================

def run_final_model():

    X, y, feature_names = load_features(
        MASTER_PATH
    )

    labels = np.unique(y)

    print("\n")
    print("=" * 70)
    print("DATASET INFORMATION")
    print("=" * 70)

    print("Task:", TASK)
    print("Detected:", DETECTED)
    print("Model:", MODEL_NAME)
    print("PCA Components:", PCA_COMPONENTS)
    print("Master Path:", MASTER_PATH)
    print("X Shape:", X.shape)
    print("Number of Samples:", X.shape[0])
    print("Number of Raw Features:", X.shape[1])
    print("Classes:", labels)

    pipeline = get_pipeline(
        model_name=MODEL_NAME,
        seed=RANDOM_SEED,
        pca_components=PCA_COMPONENTS
    )

    cv = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_SEED
    )

    # =====================================================
    # CROSS-VALIDATED PREDICTION
    # =====================================================

    print("\n")
    print("=" * 70)
    print("CROSS-VALIDATION EVALUATION")
    print("=" * 70)

    y_pred_cv = cross_val_predict(
        pipeline,
        X,
        y,
        cv=cv
    )

    metrics = calculate_metrics(
        y_true=y,
        y_pred=y_pred_cv,
        labels=labels
    )

    metrics_df = pd.DataFrame({
        "Metric": list(metrics.keys()),
        "Score (%)": list(metrics.values())
    })

    print(metrics_df.round(3))

    report_df = pd.DataFrame(
        classification_report(
            y,
            y_pred_cv,
            labels=labels,
            output_dict=True,
            zero_division=0
        )
    ).transpose()

    cm = confusion_matrix(
        y,
        y_pred_cv,
        labels=labels
    )

    cm_df = pd.DataFrame(
        cm,
        index=["True_" + str(c) for c in labels],
        columns=["Pred_" + str(c) for c in labels]
    )

    normalized_cm = confusion_matrix(
        y,
        y_pred_cv,
        labels=labels,
        normalize="true"
    )

    normalized_cm_df = pd.DataFrame(
        normalized_cm,
        index=["True_" + str(c) for c in labels],
        columns=["Pred_" + str(c) for c in labels]
    )

    # =====================================================
    # FOLD-BY-FOLD EVALUATION
    # =====================================================

    fold_results = []

    fold = 1

    for tr, te in cv.split(X, y):

        Xtr = X[tr]
        ytr = y[tr]

        Xte = X[te]
        yte = y[te]

        fold_pipeline = get_pipeline(
            model_name=MODEL_NAME,
            seed=RANDOM_SEED,
            pca_components=PCA_COMPONENTS
        )

        fold_pipeline.fit(
            Xtr,
            ytr
        )

        pred = fold_pipeline.predict(
            Xte
        )

        fold_metrics = calculate_metrics(
            y_true=yte,
            y_pred=pred,
            labels=labels
        )

        row = {
            "Fold": fold
        }

        row.update(
            fold_metrics
        )

        fold_results.append(
            row
        )

        fold += 1

    fold_results_df = pd.DataFrame(
        fold_results
    )

    fold_summary_df = pd.DataFrame({
        "Metric": [
            "Accuracy",
            "Precision Macro",
            "Recall Macro",
            "F1 Macro",
            "Precision Weighted",
            "Recall Weighted",
            "F1 Weighted",
            "Specificity",
            "Balanced Accuracy"
        ],
        "Mean (%)": [
            fold_results_df["Accuracy"].mean(),
            fold_results_df["Precision Macro"].mean(),
            fold_results_df["Recall Macro"].mean(),
            fold_results_df["F1 Macro"].mean(),
            fold_results_df["Precision Weighted"].mean(),
            fold_results_df["Recall Weighted"].mean(),
            fold_results_df["F1 Weighted"].mean(),
            fold_results_df["Specificity"].mean(),
            fold_results_df["Balanced Accuracy"].mean()
        ],
        "STD (%)": [
            fold_results_df["Accuracy"].std(),
            fold_results_df["Precision Macro"].std(),
            fold_results_df["Recall Macro"].std(),
            fold_results_df["F1 Macro"].std(),
            fold_results_df["Precision Weighted"].std(),
            fold_results_df["Recall Weighted"].std(),
            fold_results_df["F1 Weighted"].std(),
            fold_results_df["Specificity"].std(),
            fold_results_df["Balanced Accuracy"].std()
        ]
    })

    print("\n")
    print("=" * 70)
    print("FOLD SUMMARY")
    print("=" * 70)

    print(fold_summary_df.round(3))

    # =====================================================
    # TRAIN FINAL MODEL ON ALL DATA
    # =====================================================

    print("\n")
    print("=" * 70)
    print("TRAIN FINAL MODEL ON ALL DATA")
    print("=" * 70)

    final_pipeline = get_pipeline(
        model_name=MODEL_NAME,
        seed=RANDOM_SEED,
        pca_components=PCA_COMPONENTS
    )

    final_pipeline.fit(
        X,
        y
    )

    pca_variance_df = plot_pca_explained_variance(
        final_pipeline=final_pipeline
    )

    # =====================================================
    # PLOTS
    # =====================================================

    plot_main_metrics_bar(
        metrics_df=metrics_df
    )

    plot_all_metrics_bar(
        metrics_df=metrics_df
    )

    plot_confusion_matrix(
        y_true=y,
        y_pred=y_pred_cv,
        labels=labels
    )

    plot_normalized_confusion_matrix(
        y_true=y,
        y_pred=y_pred_cv,
        labels=labels
    )

    plot_cv_fold_metrics(
        fold_results_df=fold_results_df
    )

    # =====================================================
    # SAVE FINAL MODEL
    # =====================================================

    if SAVE_FINAL_MODEL:

        model_package = {
            "pipeline": final_pipeline,
            "task": TASK,
            "detected": DETECTED,
            "model_name": MODEL_NAME,
            "pca_components": PCA_COMPONENTS,
            "feature_names": feature_names,
            "labels": labels,
            "n_raw_features": X.shape[1],
            "n_samples": X.shape[0],
            "random_seed": RANDOM_SEED
        }

        joblib.dump(
            model_package,
            FINAL_MODEL_PATH
        )

        print("Final model saved at:")
        print(os.path.abspath(FINAL_MODEL_PATH))

    # =====================================================
    # SAVE EXCEL
    # =====================================================

    settings_df = pd.DataFrame({
        "Setting": [
            "Task",
            "Detected",
            "Model",
            "PCA Components",
            "N Splits",
            "Random Seed",
            "Number of Samples",
            "Number of Raw Features",
            "Master Path"
        ],
        "Value": [
            TASK,
            DETECTED,
            MODEL_NAME,
            PCA_COMPONENTS,
            N_SPLITS,
            RANDOM_SEED,
            X.shape[0],
            X.shape[1],
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

        metrics_df.to_excel(
            writer,
            sheet_name="CV_Metrics",
            index=False
        )

        fold_summary_df.to_excel(
            writer,
            sheet_name="Fold_Summary",
            index=False
        )

        fold_results_df.to_excel(
            writer,
            sheet_name="All_Folds",
            index=False
        )

        report_df.to_excel(
            writer,
            sheet_name="Classification_Report"
        )

        cm_df.to_excel(
            writer,
            sheet_name="Confusion_Matrix"
        )

        normalized_cm_df.to_excel(
            writer,
            sheet_name="Normalized_CM"
        )

        pca_variance_df.to_excel(
            writer,
            sheet_name="PCA_Variance",
            index=False
        )

        feature_names_df.to_excel(
            writer,
            sheet_name="Raw_Features",
            index=False
        )

    print("\n")
    print("=" * 70)
    print("SAVED OUTPUTS")
    print("=" * 70)

    print("Excel:")
    print(os.path.abspath(OUTPUT_EXCEL))

    print("\nFigures:")
    print(os.path.abspath(FIGURES_FOLDER))

    print("\nResults Folder:")
    print(os.path.abspath(RESULTS_FOLDER))


# =====================================================
# LOAD SAVED MODEL AND PREDICT
# =====================================================

def load_saved_model_and_predict(model_path, new_X):

    package = joblib.load(
        model_path
    )

    pipeline = package["pipeline"]

    prediction = pipeline.predict(
        new_X
    )

    return prediction


# =====================================================
# RUN
# =====================================================

if __name__ == "__main__":

    run_final_model()