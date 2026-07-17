from pathlib import Path
import re
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# =====================================================
# SETTINGS
# =====================================================

# Write folder names exactly as they exist in Final/Data and Results.
TASKS = ["Stand", "Sit_To_Stand", "Jump", "Rise_Head"]

# The script will run Normal results and, when available, DetectedAction results.
PROCESS_NORMAL = True
PROCESS_DETECTED_ACTION = True

PCA_VARIANCE = 0.95
N_SPLITS = 5
RANDOM_SEED = 42
RF_TREES = 500
KNN_NEIGHBORS = 5

TOP_CORR_FEATURES = 20
TOP_IMPORTANCE_FEATURES = 20
TOP_CENTROID_FEATURES = 15

TASK_TITLE_FA: Dict[str, str] = {
    "Stand": "ایستادن",
    "Sit_To_Stand": "بلند شدن از صندلی",
    "Jump": "پریدن",
    "Head_Raise": "بالا آوردن سر",
    "Climb_Box": "بالا رفتن از جعبه",
    "Step_Box": "بالا رفتن از جعبه",
}

MODE_TITLE_FA = {
    "Normal": "داده‌های اصلی",
    "DetectedAction": "داده‌های جداسازی‌شده با تشخیص فعالیت",
}

sns.set_theme(style="whitegrid", context="talk")
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["axes.titlesize"] = 14
plt.rcParams["axes.labelsize"] = 12
plt.rcParams["legend.fontsize"] = 10


@dataclass
class Paths:
    project_root: Path
    results_root: Path
    report_images_root: Path
    tex_out: Path


@dataclass
class TaskPaths:
    base: Paths
    task: str
    mode: str
    task_data_root: Path
    results_folder: Path
    master_features: Path
    pca_interpretation: Path
    sample_file: Optional[Path]
    results_no_pca: Optional[Path]
    results_pca95: Optional[Path]
    find_components_excel: Optional[Path]
    compare_components_excel: Optional[Path]
    report_out: Path
    results_out: Path


@dataclass
class FigureItem:
    task: str
    mode: str
    path: str
    caption: str
    label: str
    explanation: str


# =====================================================
# BASIC HELPERS
# =====================================================

def slug(text: str) -> str:
    text = text.replace(" ", "_").replace("-", "_")
    return re.sub(r"[^A-Za-z0-9_]+", "", text)


def task_title(task: str) -> str:
    return TASK_TITLE_FA.get(task, task.replace("_", " "))


def mode_title(mode: str) -> str:
    return MODE_TITLE_FA.get(mode, mode)


def mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def first_existing(paths: List[Path]) -> Optional[Path]:
    for path in paths:
        if path.exists():
            return path
    return None


def infer_paths(project_root: Path) -> Paths:
    results_root = first_existing([
        project_root / "Results",
        project_root / "Python" / "Results",
    ]) or project_root / "Results"

    return Paths(
        project_root=project_root,
        results_root=results_root,
        report_images_root=project_root / "Report" / "Images" / "Generated" / "TaskResults",
        tex_out=project_root / "Report" / "Chapters" / "generated_plot_figures.tex",
    )


def find_sample_file(task_data_root: Path, task: str) -> Optional[Path]:
    patterns = [
        f"Filtered.{task}_*.xlsx",
        f"Filtered_{task}_*.xlsx",
        f"Filtered*{task}*.xlsx",
        "Filtered*.xlsx",
        "*.xlsx",
    ]
    for pattern in patterns:
        files = sorted(task_data_root.glob(pattern))
        files = [p for p in files if "Master Features" not in str(p)]
        if files:
            return files[0]
    return None


def infer_task_paths(base: Paths, task: str, mode: str) -> Optional[TaskPaths]:
    if mode == "DetectedAction":
        task_data_root = base.project_root / "Final" / "DetectedActionData" / task
        results_folder = base.results_root / task / "DetectedAction"
    else:
        task_data_root = base.project_root / "Final" / "Data" / task
        results_folder = base.results_root / task / "Normal"

    master_dir = task_data_root / "Master Features"
    master_features = master_dir / f"MASTER_Features_{task}.xlsx"

    if not master_features.exists() and not results_folder.exists():
        warnings.warn(f"Skipped {task}/{mode}: no master features or result folder found.")
        return None

    results_no_pca = first_existing([
        results_folder / f"Results_Comparison_{task}.xlsx",
        results_folder / f"Results_Comparison_Detected_Action_{task}.xlsx",
    ])

    results_pca95 = first_existing([
        results_folder / f"Results_Comparison_Fixed_PCA95_{task}.xlsx",
        results_folder / f"Results_Comparison_Pipeline_PCA95_{task}.xlsx",
        results_folder / f"Results_Comparison_PipelinePCA95_{task}.xlsx",
        results_folder / f"Results_Comparison_Fixed_PCA95_Detected_Action_{task}.xlsx",
        results_folder / f"Results_Comparison_Pipeline_PCA95_Detected_Action_{task}.xlsx",
    ])

    find_components_excel = first_existing([
        results_folder / f"FindPrincipalComponents_PipelinePCA_{task}" / "PCA_Dimension_Search_PipelinePCA.xlsx",
        results_folder / f"FindPrincipalComponents_{task}" / "PCA_Dimension_Search_PipelinePCA.xlsx",
    ])

    compare_components_excel = first_existing([
        results_folder / f"CompareNumberOfPrincipalComponents_PipelinePCA_{task}" / "Results_All_PipelinePCA_Components.xlsx",
        results_folder / f"CompareNumberOfPrincipalComponents_{task}" / "Results_All_PipelinePCA_Components.xlsx",
    ])

    return TaskPaths(
        base=base,
        task=task,
        mode=mode,
        task_data_root=task_data_root,
        results_folder=results_folder,
        master_features=master_features,
        pca_interpretation=master_dir / "PCA_Interpretation_95.xlsx",
        sample_file=find_sample_file(task_data_root, task),
        results_no_pca=results_no_pca,
        results_pca95=results_pca95,
        find_components_excel=find_components_excel,
        compare_components_excel=compare_components_excel,
        report_out=base.report_images_root / task / mode,
        results_out=results_folder / "Generated_Plots",
    )


def read_excel(path: Optional[Path], sheet_name: Optional[str] = None) -> Optional[pd.DataFrame]:
    if path is None or not path.exists():
        return None
    try:
        if sheet_name is None:
            return pd.read_excel(path)
        return pd.read_excel(path, sheet_name=sheet_name)
    except ValueError:
        return None
    except Exception as exc:
        warnings.warn(f"Could not read {path}: {exc}")
        return None


def numeric_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.replace([np.inf, -np.inf], np.nan)
    return out.fillna(out.median(numeric_only=True))


def load_master(master_path: Path) -> Tuple[Optional[pd.DataFrame], Optional[np.ndarray], Optional[np.ndarray], List[str]]:
    df = read_excel(master_path)
    if df is None or "label" not in df.columns:
        return df, None, None, []

    meta_cols = ["label", "source_file", "prefix", "idx"]
    feature_cols = [col for col in df.columns if col not in meta_cols]
    X_df = numeric_df(df[feature_cols])
    y = df["label"].astype(int).to_numpy()
    return df, X_df.to_numpy(dtype=float), y, feature_cols


def save_fig(fig: Figure, tp: TaskPaths, filename: str) -> str:
    mkdir(tp.report_out)
    mkdir(tp.results_out)
    report_path = tp.report_out / filename
    results_path = tp.results_out / filename
    fig.tight_layout()
    fig.savefig(report_path, bbox_inches="tight")
    fig.savefig(results_path, bbox_inches="tight")
    plt.close(fig)
    return report_path.relative_to(tp.base.project_root / "Report").as_posix()


def add_fig(figs: List[FigureItem], tp: TaskPaths, rel_path: str, caption: str, label_suffix: str, explanation: str) -> None:
    figs.append(FigureItem(
        task=tp.task,
        mode=tp.mode,
        path=rel_path,
        caption=caption,
        label=f"fig:{slug(tp.task).lower()}_{slug(tp.mode).lower()}_{label_suffix}",
        explanation=explanation,
    ))


# =====================================================
# MODELS
# =====================================================

def get_models(use_pca: bool) -> Dict[str, Pipeline]:
    if use_pca:
        return {
            "KNN": Pipeline([
                ("scaler", StandardScaler()),
                ("pca", PCA(n_components=PCA_VARIANCE, random_state=RANDOM_SEED)),
                ("clf", KNeighborsClassifier(n_neighbors=KNN_NEIGHBORS)),
            ]),
            "SVM": Pipeline([
                ("scaler", StandardScaler()),
                ("pca", PCA(n_components=PCA_VARIANCE, random_state=RANDOM_SEED)),
                ("clf", SVC(kernel="rbf")),
            ]),
            "Random Forest": Pipeline([
                ("scaler", StandardScaler()),
                ("pca", PCA(n_components=PCA_VARIANCE, random_state=RANDOM_SEED)),
                ("clf", RandomForestClassifier(n_estimators=RF_TREES, random_state=RANDOM_SEED)),
            ]),
        }

    return {
        "KNN": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier(n_neighbors=KNN_NEIGHBORS)),
        ]),
        "SVM": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf")),
        ]),
        "Random Forest": Pipeline([
            ("clf", RandomForestClassifier(n_estimators=RF_TREES, random_state=RANDOM_SEED)),
        ]),
    }


# =====================================================
# MASTER DATA PLOTS
# =====================================================

def plot_signal_sample(tp: TaskPaths, figs: List[FigureItem]) -> None:
    df = read_excel(tp.sample_file)
    if df is None:
        return

    nums = numeric_df(df)
    if nums.empty:
        return

    preferred = ["Head Ax", "head_ax", "Right Hand Ax", "right_hand_ax", "Left Hand Ax", "left_hand_ax", "Right Foot Ax", "right_foot_ax", "Left Foot Ax", "left_foot_ax"]
    selected: List[str] = []
    for key in preferred:
        found = next((c for c in nums.columns if key.lower() in str(c).lower()), None)
        if found is not None and found not in selected:
            selected.append(found)
        if len(selected) >= 5:
            break

    if not selected:
        selected = nums.var(numeric_only=True).sort_values(ascending=False).head(3).index.tolist()
    if not selected:
        return

    fig, ax = plt.subplots(figsize=(12, 5))
    for col in selected:
        ax.plot(nums.index, nums[col], label=str(col), linewidth=1.1)
    ax.set_title(f"Sample Sensor Signals - {tp.task} ({tp.mode})")
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Signal Value")
    ax.legend(loc="best")

    rel = save_fig(fig, tp, "signal_sample.png")
    add_fig(figs, tp, rel,
            f"نمونه سیگنال‌های ثبت‌شده در فعالیت {task_title(tp.task)} ({mode_title(tp.mode)}).",
            "signal_sample",
            "این شکل نمایی از ماهیت زمانی سیگنال‌های حسگرها را برای فعالیت مورد نظر نشان می‌دهد.")


def plot_class_distribution(tp: TaskPaths, figs: List[FigureItem]) -> None:
    _, _, y, _ = load_master(tp.master_features)
    if y is None:
        return

    counts = pd.Series(y).value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x=counts.index.astype(str), y=counts.values, ax=ax)
    ax.set_title(f"Class Distribution - {tp.task} ({tp.mode})")
    ax.set_xlabel("Label")
    ax.set_ylabel("Number of Samples")
    for i, value in enumerate(counts.values):
        ax.text(i, value + 0.2, str(value), ha="center", va="bottom", fontsize=10)

    rel = save_fig(fig, tp, "class_distribution.png")
    add_fig(figs, tp, rel,
            f"توزیع تعداد نمونه‌ها در سطوح عملکردی فعالیت {task_title(tp.task)}.",
            "class_distribution",
            "این نمودار برای بررسی تعادل داده‌ها بین کلاس‌های صفر، یک و دو استفاده می‌شود.")


def plot_feature_correlation(tp: TaskPaths, figs: List[FigureItem]) -> None:
    df, _, _, feature_cols = load_master(tp.master_features)
    if df is None or not feature_cols:
        return

    nums = numeric_df(df[feature_cols])
    top = nums.var(numeric_only=True).sort_values(ascending=False).head(TOP_CORR_FEATURES).index.tolist()
    if len(top) < 2:
        return

    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(nums[top].corr(), ax=ax, cmap="coolwarm", center=0, square=True)
    ax.set_title(f"Feature Correlation - {tp.task} ({tp.mode})")

    rel = save_fig(fig, tp, "feature_correlation_heatmap.png")
    add_fig(figs, tp, rel,
            f"نقشه همبستگی {TOP_CORR_FEATURES} ویژگی با بیشترین واریانس در فعالیت {task_title(tp.task)}.",
            "feature_correlation",
            "این نمودار وجود اطلاعات تکراری در فضای ویژگی را نشان می‌دهد و استفاده از کاهش بعد را توجیه می‌کند.")


def compute_pca_from_master(tp: TaskPaths) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    _, X, y, _ = load_master(tp.master_features)
    if X is None or y is None:
        return None, None

    X_scaled = StandardScaler().fit_transform(X)
    max_components = min(X_scaled.shape[0] - 1, X_scaled.shape[1])
    if max_components < 2:
        return None, None

    pca = PCA(n_components=max_components, random_state=RANDOM_SEED)
    scores = pca.fit_transform(X_scaled)

    variance = pd.DataFrame({
        "PC": [f"PC{i + 1}" for i in range(len(pca.explained_variance_ratio_))],
        "Explained_Variance_%": pca.explained_variance_ratio_ * 100,
        "Cumulative_Variance_%": np.cumsum(pca.explained_variance_ratio_) * 100,
    })
    scores_df = pd.DataFrame({"PC1": scores[:, 0], "PC2": scores[:, 1], "label": y})
    return variance, scores_df


def plot_pca_variance_and_scatter(tp: TaskPaths, figs: List[FigureItem]) -> None:
    variance = read_excel(tp.pca_interpretation, "Variance")
    scores = read_excel(tp.pca_interpretation, "PC_Scores")
    if variance is None or not {"Explained_Variance_%", "Cumulative_Variance_%"}.issubset(variance.columns):
        variance, scores = compute_pca_from_master(tp)
    if variance is None:
        return

    x = np.arange(1, len(variance) + 1)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x, variance["Explained_Variance_%"], alpha=0.7, label="Explained Variance (%)")
    ax.plot(x, variance["Cumulative_Variance_%"], linewidth=2.5, label="Cumulative Variance (%)")
    ax.axhline(95, linestyle="--", linewidth=1.2, label="95% Threshold")
    ax.set_title(f"PCA Explained Variance - {tp.task} ({tp.mode})")
    ax.set_xlabel("Principal Component Index")
    ax.set_ylabel("Variance (%)")
    ax.legend(loc="best")

    rel = save_fig(fig, tp, "pca_variance_curve.png")
    add_fig(figs, tp, rel,
            f"درصد واریانس توضیح‌داده‌شده و واریانس تجمعی مؤلفه‌های اصلی در فعالیت {task_title(tp.task)}.",
            "pca_variance",
            "این شکل مبنای انتخاب آستانه ۹۵ درصد و تعیین تعداد مؤلفه‌های نگه‌داشته‌شده را نشان می‌دهد.")

    if scores is None or not {"PC1", "PC2", "label"}.issubset(scores.columns):
        _, scores = compute_pca_from_master(tp)
    if scores is None:
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(data=scores, x="PC1", y="PC2", hue="label", s=60, ax=ax)
    ax.set_title(f"PCA Projection - {tp.task} ({tp.mode})")

    rel = save_fig(fig, tp, "pca_scatter_pc1_pc2.png")
    add_fig(figs, tp, rel,
            f"پراکندگی نمونه‌های فعالیت {task_title(tp.task)} در فضای دو مؤلفه اصلی اول.",
            "pca_scatter",
            "این شکل یک تصویر اولیه از جدایی یا هم‌پوشانی کلاس‌ها در فضای کاهش‌یافته ارائه می‌دهد.")


def plot_rf_feature_importance_and_centroids(tp: TaskPaths, figs: List[FigureItem]) -> None:
    df, X, y, feature_cols = load_master(tp.master_features)
    if df is None or X is None or y is None or not feature_cols:
        return

    rf = RandomForestClassifier(n_estimators=RF_TREES, random_state=RANDOM_SEED)
    rf.fit(X, y)
    imp = pd.DataFrame({"Feature": feature_cols, "Importance": rf.feature_importances_})
    imp = imp.sort_values("Importance", ascending=False).head(TOP_IMPORTANCE_FEATURES)

    fig, ax = plt.subplots(figsize=(10, 7))
    sns.barplot(data=imp, x="Importance", y="Feature", ax=ax)
    ax.set_title(f"Top Feature Importances - {tp.task} ({tp.mode})")

    rel = save_fig(fig, tp, "rf_top_feature_importance.png")
    add_fig(figs, tp, rel,
            f"{TOP_IMPORTANCE_FEATURES} ویژگی برتر بر اساس اهمیت در مدل \\lr{{Random Forest}} برای فعالیت {task_title(tp.task)}.",
            "rf_importance",
            "این شکل نشان می‌دهد کدام حسگرها، محورها و شاخص‌های آماری نقش بیشتری در تصمیم‌گیری مدل داشته‌اند.")

    nums = numeric_df(df[feature_cols])
    top = nums.var(numeric_only=True).sort_values(ascending=False).head(TOP_CENTROID_FEATURES).index.tolist()
    if not top:
        return
    z = (nums[top] - nums[top].mean()) / (nums[top].std(ddof=0) + 1e-9)
    class_z = z.groupby(df["label"]).mean()

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(class_z, cmap="vlag", center=0, ax=ax)
    ax.set_title(f"Class-wise Mean Z-Score - {tp.task} ({tp.mode})")
    ax.set_xlabel("Feature")
    ax.set_ylabel("Label")

    rel = save_fig(fig, tp, "class_centroid_feature_heatmap.png")
    add_fig(figs, tp, rel,
            f"میانگین نرمال‌شده ویژگی‌های برتر در کلاس‌های مختلف فعالیت {task_title(tp.task)}.",
            "class_centroid_heatmap",
            "این نقشه حرارتی الگوی میانگین ویژگی‌ها را در هر سطح عملکردی نمایش می‌دهد.")


# =====================================================
# RESULT EXCEL PLOTS
# =====================================================

def read_summary(path: Optional[Path]) -> Optional[pd.DataFrame]:
    df = read_excel(path, "Summary")
    if df is None:
        df = read_excel(path)
    if df is None or "Model" not in df.columns:
        return None
    return df


def read_folds(path: Optional[Path]) -> Optional[pd.DataFrame]:
    for sheet in ["All_Folds", "All_Runs"]:
        df = read_excel(path, sheet)
        if df is not None:
            return df
    return None


def plot_model_result_summaries(tp: TaskPaths, figs: List[FigureItem]) -> None:
    no_pca = read_summary(tp.results_no_pca)
    pca95 = read_summary(tp.results_pca95)
    if no_pca is None and pca95 is None:
        return

    frames = []
    if no_pca is not None:
        tmp = no_pca.copy()
        tmp["Setup"] = "No PCA"
        frames.append(tmp)
    if pca95 is not None:
        tmp = pca95.copy()
        tmp["Setup"] = "PCA95"
        frames.append(tmp)

    combo = pd.concat(frames, ignore_index=True)
    metric = "F1" if "F1" in combo.columns else "Accuracy"
    if metric not in combo.columns:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=combo, x="Model", y=metric, hue="Setup", ax=ax)
    ax.set_title(f"Model Comparison by {metric} - {tp.task} ({tp.mode})")
    ax.set_ylabel(f"{metric} (%)")
    ax.set_xlabel("Model")
    ax.set_ylim(0, 100)

    rel = save_fig(fig, tp, f"model_comparison_{metric.lower()}.png")
    add_fig(figs, tp, rel,
            f"مقایسه عملکرد مدل‌ها بر اساس معیار \\lr{{{metric}}} در فعالیت {task_title(tp.task)}.",
            f"model_comparison_{metric.lower()}",
            "این نمودار اثر استفاده یا عدم استفاده از کاهش بعد را برای هر مدل نشان می‌دهد.")

    ranking_source = no_pca if no_pca is not None else pca95
    ranking = ranking_source.sort_values(metric, ascending=False)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=ranking, x="Model", y=metric, ax=ax)
    ax.set_title(f"Model Ranking by {metric} - {tp.task} ({tp.mode})")
    ax.set_ylabel(f"{metric} (%)")
    ax.set_ylim(0, 100)

    rel = save_fig(fig, tp, f"model_ranking_{metric.lower()}.png")
    add_fig(figs, tp, rel,
            f"رتبه‌بندی مدل‌ها بر اساس معیار \\lr{{{metric}}} در فعالیت {task_title(tp.task)}.",
            f"model_ranking_{metric.lower()}",
            "این شکل جمع‌بندی سریعی از برتری نسبی مدل‌ها ارائه می‌دهد.")

    if no_pca is not None and pca95 is not None and {"Model", "F1"}.issubset(no_pca.columns) and {"Model", "F1"}.issubset(pca95.columns):
        merged = no_pca[["Model", "F1"]].merge(pca95[["Model", "F1"]], on="Model", suffixes=("_NoPCA", "_PCA95"))
        merged["F1_Gain"] = merged["F1_PCA95"] - merged["F1_NoPCA"]
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(data=merged, x="Model", y="F1_Gain", ax=ax)
        ax.axhline(0, linewidth=1)
        ax.set_title(f"F1 Gain from PCA95 - {tp.task} ({tp.mode})")
        ax.set_ylabel("F1 Difference (percentage points)")

        rel = save_fig(fig, tp, "f1_gain_pca_vs_no_pca.png")
        add_fig(figs, tp, rel,
                f"تغییر معیار \\lr{{F1}} پس از اعمال \\lr{{PCA95}} در فعالیت {task_title(tp.task)}.",
                "f1_gain_pca",
                "این شکل نشان می‌دهد کاهش بعد برای هر مدل سودمند بوده یا موجب افت عملکرد شده است.")

    folds = read_folds(tp.results_no_pca)
    if folds is None:
        folds = read_folds(tp.results_pca95)
    if folds is not None and "Model" in folds.columns:
        metrics = [m for m in ["Accuracy", "Precision", "Recall", "Specificity", "F1"] if m in folds.columns]
        if metrics:
            long_df = folds.melt(id_vars=["Model"], value_vars=metrics, var_name="Metric", value_name="Score")
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.boxplot(data=long_df, x="Metric", y="Score", hue="Model", ax=ax)
            ax.set_title(f"Metric Distribution Across Folds - {tp.task} ({tp.mode})")
            ax.set_ylim(0, 100)

            rel = save_fig(fig, tp, "metric_boxplots.png")
            add_fig(figs, tp, rel,
                    f"پراکندگی معیارهای ارزیابی در تکرارهای اعتبارسنجی برای فعالیت {task_title(tp.task)}.",
                    "metric_boxplots",
                    "این شکل پایداری مدل‌ها را در foldهای مختلف نشان می‌دهد.")


def plot_cv_confusion_and_recall(tp: TaskPaths, figs: List[FigureItem], use_pca: bool) -> None:
    _, X, y, _ = load_master(tp.master_features)
    if X is None or y is None:
        return

    models = get_models(use_pca)
    labels = np.unique(y)
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)
    setup = "pca95" if use_pca else "no_pca"

    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 4.5))
    axes = np.atleast_1d(axes)
    recall_rows = []

    for ax, (name, model) in zip(axes, models.items()):
        pred = cross_val_predict(model, X, y, cv=cv)
        cm = confusion_matrix(y, pred, labels=labels, normalize="true")
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        disp.plot(ax=ax, colorbar=False, cmap="Blues", values_format=".2f")
        ax.set_title(name)
        for i, label in enumerate(labels):
            recall_rows.append({"Model": name, "Label": str(label), "Recall": cm[i, i] * 100})

    fig.suptitle(f"Normalized Confusion Matrices - {tp.task} ({setup})", y=1.03)
    rel = save_fig(fig, tp, f"confusion_matrices_{setup}.png")
    add_fig(figs, tp, rel,
            f"ماتریس درهم‌ریختگی نرمال‌شده مدل‌ها در فعالیت {task_title(tp.task)} ({setup}).",
            f"confusion_{setup}",
            "این شکل مشخص می‌کند خطاهای اصلی بین کدام کلاس‌های عملکردی رخ داده است.")

    recall_df = pd.DataFrame(recall_rows)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=recall_df, x="Label", y="Recall", hue="Model", ax=ax)
    ax.set_ylim(0, 100)
    ax.set_title(f"Per-Class Recall - {tp.task} ({setup})")

    rel = save_fig(fig, tp, f"per_class_recall_{setup}.png")
    add_fig(figs, tp, rel,
            f"مقایسه \\lr{{Recall}} هر کلاس در فعالیت {task_title(tp.task)} ({setup}).",
            f"per_class_recall_{setup}",
            "این نمودار حساسیت مدل‌ها نسبت به هر سطح عملکردی را نشان می‌دهد.")


def plot_component_search_results(tp: TaskPaths, figs: List[FigureItem]) -> None:
    df = read_excel(tp.find_components_excel, "Results")
    if df is not None and {"Components", "Accuracy"}.issubset(df.columns):
        metrics = [m for m in ["Accuracy", "Precision", "Recall", "Specificity", "F1", "Balanced Accuracy"] if m in df.columns]
        fig, ax = plt.subplots(figsize=(11, 7))
        for metric in metrics:
            ax.plot(df["Components"], df[metric], linewidth=2, marker="o", markersize=4, label=metric)
        ax.invert_xaxis()
        ax.set_xlabel("Number of Principal Components")
        ax.set_ylabel("Score (%)")
        ax.set_ylim(0, 100)
        ax.set_title(f"SVM Performance vs PCA Components - {tp.task} ({tp.mode})")
        ax.legend(loc="best")

        rel = save_fig(fig, tp, "svm_metrics_vs_pca_components.png")
        add_fig(figs, tp, rel,
                f"تغییر عملکرد مدل \\lr{{SVM}} نسبت به تعداد مؤلفه‌های اصلی در فعالیت {task_title(tp.task)}.",
                "svm_metrics_vs_pca_components",
                "این نمودار برای انتخاب بازه مناسب تعداد مؤلفه‌های اصلی استفاده می‌شود.")

    cmp_df = read_excel(tp.compare_components_excel, "Results")
    if cmp_df is not None and {"Components", "Model"}.issubset(cmp_df.columns):
        metric = "F1" if "F1" in cmp_df.columns else "Accuracy"
        if metric not in cmp_df.columns:
            return
        fig, ax = plt.subplots(figsize=(10, 6))
        for model in cmp_df["Model"].dropna().unique():
            temp = cmp_df[cmp_df["Model"] == model]
            ax.plot(temp["Components"], temp[metric], linewidth=2, marker="o", markersize=4, label=model)
        ax.set_xlabel("Number of PCA Components")
        ax.set_ylabel(f"{metric} (%)")
        ax.set_ylim(0, 100)
        ax.set_title(f"Model Comparison vs PCA Components - {tp.task} ({tp.mode})")
        ax.legend(loc="best")

        rel = save_fig(fig, tp, f"models_{metric.lower()}_vs_pca_components.png")
        add_fig(figs, tp, rel,
                f"مقایسه مدل‌ها نسبت به تعداد مؤلفه‌های اصلی در فعالیت {task_title(tp.task)}.",
                f"models_{metric.lower()}_vs_pca_components",
                "این نمودار نشان می‌دهد انتخاب تعداد مؤلفه‌ها و انتخاب مدل باید هم‌زمان انجام شود.")


def build_task(tp: TaskPaths, figs: List[FigureItem]) -> None:
    plot_signal_sample(tp, figs)
    plot_class_distribution(tp, figs)
    plot_feature_correlation(tp, figs)
    plot_pca_variance_and_scatter(tp, figs)
    plot_model_result_summaries(tp, figs)
    plot_cv_confusion_and_recall(tp, figs, use_pca=False)
    plot_cv_confusion_and_recall(tp, figs, use_pca=True)
    plot_rf_feature_importance_and_centroids(tp, figs)
    plot_component_search_results(tp, figs)


# =====================================================
# LATEX OUTPUT
# =====================================================

def write_latex_guide(base: Paths, figs: List[FigureItem]) -> None:
    mkdir(base.tex_out.parent)
    lines: List[str] = []
    lines.append("% Auto-generated by Python/Plot/plot_report_figures.py")
    lines.append("% Paste this file with: \\input{Chapters/generated_plot_figures}")
    lines.append("")

    for task in TASKS:
        task_figs = [f for f in figs if f.task == task]
        if not task_figs:
            continue
        lines.append(f"\\subsection*{{نمودارهای فعالیت {task_title(task)}}}")
        lines.append("")
        for mode in ["Normal", "DetectedAction"]:
            mode_figs = [f for f in task_figs if f.mode == mode]
            if not mode_figs:
                continue
            lines.append(f"\\subsubsection*{{{mode_title(mode)}}}")
            lines.append("")
            for item in mode_figs:
                lines.append("\\begin{figure}[H]")
                lines.append("    \\centering")
                lines.append(f"    \\includegraphics[width=0.85\\linewidth]{{{item.path}}}")
                lines.append(f"    \\caption{{{item.caption}}}")
                lines.append(f"    \\label{{{item.label}}}")
                lines.append("\\end{figure}")
                lines.append("")
                lines.append("\\noindent\\textbf{توضیح شکل:} " + item.explanation)
                lines.append("\\vspace{0.8em}")
                lines.append("")

    base.tex_out.write_text("\n".join(lines), encoding="utf-8")


def build_all(project_root: Path) -> List[FigureItem]:
    base = infer_paths(project_root)
    figs: List[FigureItem] = []
    modes: List[str] = []
    if PROCESS_NORMAL:
        modes.append("Normal")
    if PROCESS_DETECTED_ACTION:
        modes.append("DetectedAction")

    for task in TASKS:
        for mode in modes:
            tp = infer_task_paths(base, task, mode)
            if tp is None:
                continue
            print("\n" + "=" * 70)
            print(f"Building plots for task={task}, mode={mode}")
            print("Master:", tp.master_features)
            print("Results:", tp.results_folder)
            print("=" * 70)
            build_task(tp, figs)

    write_latex_guide(base, figs)
    return figs


if __name__ == "__main__":
    this_file = Path(__file__).resolve()
    project_root = this_file.parents[2]
    print("Project root:", project_root)
    figures = build_all(project_root)
    print("\nGenerated figures:")
    for item in figures:
        print(f"[{item.task} | {item.mode}] {item.path}")
    print("\nDone.")
