import warnings
from dataclasses import dataclass
from pathlib import Path
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


sns.set_theme(style="whitegrid", context="talk")
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["axes.titlesize"] = 14
plt.rcParams["axes.labelsize"] = 12
plt.rcParams["legend.fontsize"] = 10


@dataclass
class Paths:
    project_root: Path
    report_images_out: Path
    results_images_out: Path
    tex_out: Path
    master_features: Path
    pca_interpretation: Path
    results_no_pca: Path
    results_pca95: Path
    sample_filtered_file: Path


@dataclass
class FigureItem:
    chapter: str
    filename: str
    caption: str
    label: str
    explanation: str


def _safe_read_excel(path: Path, sheet_name: Optional[str] = None) -> Optional[pd.DataFrame]:
    if not path.exists():
        warnings.warn(f"Missing file: {path}")
        return None
    try:
        if sheet_name is None:
            return pd.read_excel(path)
        return pd.read_excel(path, sheet_name=sheet_name)
    except Exception as exc:
        warnings.warn(f"Could not read {path}: {exc}")
        return None


def _mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def infer_paths(project_root: Path) -> Paths:
    final_data = project_root / "Final" / "Data" / "Stand"
    master = final_data / "Master Features"
    python_results = project_root / "Python" / "Results"

    report_images_out = project_root / "Report" / "Images" / "Generated"
    results_images_out = python_results / "Generated_Plots"
    tex_out = project_root / "Report" / "Chapters" / "generated_plot_figures.tex"

    sample_file = final_data / "Filtered.Stand_0.01.xlsx"
    if not sample_file.exists():
        all_filtered = sorted(final_data.glob("Filtered.Stand_*.xlsx"))
        if all_filtered:
            sample_file = all_filtered[0]

    return Paths(
        project_root=project_root,
        report_images_out=report_images_out,
        results_images_out=results_images_out,
        tex_out=tex_out,
        master_features=master / "MASTER_Features_Stand.xlsx",
        pca_interpretation=master / "PCA_Interpretation_95.xlsx",
        results_no_pca=python_results / "Results_Comparison.xlsx",
        results_pca95=python_results / "Results_Comparison_PCA95.xlsx",
        sample_filtered_file=sample_file,
    )


def _save_dual(fig: Figure, chapter: str, filename: str, paths: Paths) -> Tuple[Path, Path]:
    report_path = paths.report_images_out / chapter / filename
    results_path = paths.results_images_out / chapter / filename

    _mkdir(report_path.parent)
    _mkdir(results_path.parent)

    fig.tight_layout()
    fig.savefig(report_path, bbox_inches="tight")
    fig.savefig(results_path, bbox_inches="tight")
    plt.close(fig)

    return report_path, results_path


def load_master_xy(master_path: Path) -> Tuple[Optional[pd.DataFrame], Optional[np.ndarray], Optional[np.ndarray], List[str]]:
    df = _safe_read_excel(master_path)
    if df is None:
        return None, None, None, []

    if "label" not in df.columns:
        warnings.warn(f"label column not found in {master_path}")
        return df, None, None, []

    meta_cols = ["label", "source_file", "prefix", "idx"]
    feature_cols = [c for c in df.columns if c not in meta_cols]

    X = df[feature_cols].copy()
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")

    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True))

    y = df["label"].astype(int).to_numpy()
    return df, X.to_numpy(dtype=float), y, feature_cols


def _get_models(use_pca: bool) -> Dict[str, Pipeline]:
    if use_pca:
        return {
            "KNN": Pipeline([
                ("scaler", StandardScaler()),
                ("pca", PCA(n_components=0.95, random_state=42)),
                ("clf", KNeighborsClassifier(n_neighbors=5)),
            ]),
            "SVM": Pipeline([
                ("scaler", StandardScaler()),
                ("pca", PCA(n_components=0.95, random_state=42)),
                ("clf", SVC(kernel="rbf")),
            ]),
            "Random Forest": Pipeline([
                ("scaler", StandardScaler()),
                ("pca", PCA(n_components=0.95, random_state=42)),
                ("clf", RandomForestClassifier(n_estimators=500, random_state=42)),
            ]),
        }

    return {
        "KNN": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier(n_neighbors=5)),
        ]),
        "SVM": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf")),
        ]),
        "Random Forest": Pipeline([
            ("clf", RandomForestClassifier(n_estimators=500, random_state=42)),
        ]),
    }


def ch2_signal_sample(paths: Paths, figures: List[FigureItem]) -> None:
    df = _safe_read_excel(paths.sample_filtered_file)
    if df is None:
        return

    cols = list(df.columns)
    selected = []
    for key in ["Head Ax", "Right Foot Ax", "Left Foot Ax"]:
        c = next((x for x in cols if key in str(x)), None)
        if c is not None:
            selected.append(c)

    if not selected:
        selected = cols[1:4] if len(cols) >= 4 else cols[:3]

    plot_df = df[selected].copy()
    for c in plot_df.columns:
        plot_df[c] = pd.to_numeric(plot_df[c], errors="coerce")

    fig, ax = plt.subplots(figsize=(12, 5))
    for c in plot_df.columns:
        ax.plot(plot_df.index, plot_df[c], label=str(c), linewidth=1.1)

    ax.set_title("Sample Sensor Signals (Stand Task)")
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Signal Value")
    ax.legend(loc="upper right")

    _save_dual(fig, "Chapter2", "ch2_signal_sample.png", paths)
    figures.append(FigureItem(
        chapter="Chapter2",
        filename="ch2_signal_sample.png",
        caption="نمونه سیگنال های ثبت شده از حسگرها در فعالیت ایستادن.",
        label="fig:ch2_signal_sample",
        explanation="این شکل نمایی از داده خام/فیلترشده حسگرها در یک نمونه از فعالیت ایستادن را نشان می دهد و برای معرفی ماهیت زمانی سیگنال ها استفاده می شود.",
    ))


def ch2_class_distribution(paths: Paths, figures: List[FigureItem]) -> None:
    _, _, y, _ = load_master_xy(paths.master_features)
    if y is None:
        return

    counts = pd.Series(y).value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x=counts.index.astype(str), y=counts.values, ax=ax, palette="Set2")
    ax.set_title("Class Distribution in Master Dataset")
    ax.set_xlabel("Label")
    ax.set_ylabel("Number of Samples")
    for i, v in enumerate(counts.values):
        ax.text(i, v + 0.2, str(v), ha="center", va="bottom", fontsize=10)

    _save_dual(fig, "Chapter2", "ch2_class_distribution.png", paths)
    figures.append(FigureItem(
        chapter="Chapter2",
        filename="ch2_class_distribution.png",
        caption="توزیع تعداد نمونه ها در کلاس های برچسب گذاری شده.",
        label="fig:ch2_class_distribution",
        explanation="این شکل تعادل یا عدم تعادل داده ها بین کلاس ها را نشان می دهد و برای توجیه روش ارزیابی مدل اهمیت دارد.",
    ))


def ch2_correlation_heatmap(paths: Paths, figures: List[FigureItem]) -> None:
    df, _, _, feature_cols = load_master_xy(paths.master_features)
    if df is None or not feature_cols:
        return

    num_df = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    var_series = num_df.var(numeric_only=True)
    if not isinstance(var_series, pd.Series):
        return
    top_var = var_series.sort_values(ascending=False).head(20).index.tolist()
    if len(top_var) < 2:
        return

    corr = num_df[top_var].corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, ax=ax, cmap="coolwarm", center=0, square=True)
    ax.set_title("Feature Correlation Heatmap (Top 20 by Variance)")

    _save_dual(fig, "Chapter2", "ch2_feature_correlation_heatmap.png", paths)
    figures.append(FigureItem(
        chapter="Chapter2",
        filename="ch2_feature_correlation_heatmap.png",
        caption="نقشه همبستگی 20 ویژگی با بیشترین واریانس.",
        label="fig:ch2_feature_correlation",
        explanation="این شکل میزان همبستگی بین ویژگی های منتخب را نمایش می دهد و انگیزه کاهش بعد/حذف افزونگی را تقویت می کند.",
    ))


def ch3_metric_comparison(paths: Paths, figures: List[FigureItem]) -> None:
    no_pca = _safe_read_excel(paths.results_no_pca, sheet_name="Summary")
    pca = _safe_read_excel(paths.results_pca95, sheet_name="Summary")
    if no_pca is None or pca is None:
        return

    metrics = [m for m in ["Accuracy", "Precision", "Recall", "Specificity", "F1"] if m in no_pca.columns]
    if not metrics:
        return

    no_pca["Setup"] = "No PCA"
    pca["Setup"] = "PCA95"
    combo = pd.concat([no_pca, pca], ignore_index=True)
    long_df = combo.melt(id_vars=["Model", "Setup"], value_vars=metrics, var_name="Metric", value_name="Score")

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(data=long_df, x="Metric", y="Score", hue="Setup", ci=None, ax=ax, palette="deep")
    ax.set_title("Model Performance Comparison: PCA vs No PCA")
    ax.set_ylabel("Score (%)")
    ax.set_xlabel("Metric")
    ax.set_ylim(0, 100)

    _save_dual(fig, "Chapter3", "ch3_metrics_pca_vs_no_pca.png", paths)
    figures.append(FigureItem(
        chapter="Chapter3",
        filename="ch3_metrics_pca_vs_no_pca.png",
        caption="مقایسه معیارهای عملکرد مدل ها در دو حالت بدون PCA و با PCA95.",
        label="fig:ch3_metrics_pca_vs_no_pca",
        explanation="این نمودار نشان می دهد کاهش بعد چه تاثیری بر معیارهای اصلی ارزیابی داشته است و برای جمع بندی نتایج اصلی فصل 3 مناسب است.",
    ))


def ch3_metric_boxplot(paths: Paths, figures: List[FigureItem]) -> None:
    all_runs = _safe_read_excel(paths.results_no_pca, sheet_name="All_Runs")
    if all_runs is None:
        return

    metric_cols = [m for m in ["Accuracy", "Precision", "Recall", "Specificity", "F1"] if m in all_runs.columns]
    if not metric_cols:
        return

    long_df = all_runs.melt(id_vars=["Model"], value_vars=metric_cols, var_name="Metric", value_name="Score")

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(data=long_df, x="Metric", y="Score", hue="Model", ax=ax)
    ax.set_title("Distribution of Metrics Across Repeated Runs (No PCA)")
    ax.set_ylim(0, 100)

    _save_dual(fig, "Chapter3", "ch3_metric_boxplots_no_pca.png", paths)
    figures.append(FigureItem(
        chapter="Chapter3",
        filename="ch3_metric_boxplots_no_pca.png",
        caption="پراکندگی معیارهای ارزیابی در تکرارهای مختلف (بدون PCA).",
        label="fig:ch3_metric_boxplots_no_pca",
        explanation="این شکل پایداری مدل ها را در تکرارهای تصادفی نشان می دهد و مکمل نتایج میانگین است.",
    ))


def ch3_confusion_matrices(paths: Paths, figures: List[FigureItem], use_pca: bool) -> None:
    _, X, y, _ = load_master_xy(paths.master_features)
    if X is None or y is None:
        return

    models = _get_models(use_pca=use_pca)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    labels = np.unique(y)

    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 4.5))
    if len(models) == 1:
        axes = [axes]

    for ax, (name, model) in zip(axes, models.items()):
        y_pred = cross_val_predict(model, X, y, cv=cv)
        cm = confusion_matrix(y, y_pred, labels=labels, normalize="true")
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        disp.plot(ax=ax, colorbar=False, cmap="Blues", values_format=".2f")
        ax.set_title(name)

    setup_name = "pca95" if use_pca else "no_pca"
    fig.suptitle(f"Normalized Confusion Matrices ({setup_name})", y=1.02)

    fname = f"ch3_confusion_matrices_{setup_name}.png"
    _save_dual(fig, "Chapter3", fname, paths)
    figures.append(FigureItem(
        chapter="Chapter3",
        filename=fname,
        caption=f"ماتریس درهم ریختگی نرمال شده مدل ها در حالت {setup_name}.",
        label=f"fig:ch3_confusion_{setup_name}",
        explanation="این شکل خطاهای بین کلاسی را مشخص می کند و نشان می دهد هر مدل کدام کلاس ها را بهتر یا ضعیف تر تشخیص می دهد.",
    ))


def ch3_pca_variance(paths: Paths, figures: List[FigureItem]) -> None:
    variance_df = _safe_read_excel(paths.pca_interpretation, sheet_name="Variance")
    if variance_df is None:
        return

    needed = {"PC", "Explained_Variance_%", "Cumulative_Variance_%"}
    if not needed.issubset(set(variance_df.columns)):
        return

    x = np.arange(1, len(variance_df) + 1)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x, variance_df["Explained_Variance_%"], alpha=0.7, label="Explained Variance (%)")
    ax.plot(x, variance_df["Cumulative_Variance_%"], color="crimson", linewidth=2.5, label="Cumulative Variance (%)")
    ax.axhline(95, color="black", linestyle="--", linewidth=1.2, label="95% Threshold")
    ax.set_title("PCA Explained Variance")
    ax.set_xlabel("Principal Component Index")
    ax.set_ylabel("Variance (%)")
    ax.legend(loc="best")

    _save_dual(fig, "Chapter3", "ch3_pca_variance_curve.png", paths)
    figures.append(FigureItem(
        chapter="Chapter3",
        filename="ch3_pca_variance_curve.png",
        caption="درصد واریانس توضیح داده شده و تجمعی مولفه های اصلی.",
        label="fig:ch3_pca_variance",
        explanation="این شکل مبنای انتخاب آستانه 95 درصد و تعیین تعداد مولفه های نگه داشته شده در PCA را نشان می دهد.",
    ))


def ch3_pca_scatter(paths: Paths, figures: List[FigureItem]) -> None:
    scores_df = _safe_read_excel(paths.pca_interpretation, sheet_name="PC_Scores")
    if scores_df is None or not {"PC1", "PC2", "label"}.issubset(scores_df.columns):
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(data=scores_df, x="PC1", y="PC2", hue="label", palette="tab10", s=60, ax=ax)
    ax.set_title("PCA Projection (PC1 vs PC2)")

    _save_dual(fig, "Chapter3", "ch3_pca_scatter_pc1_pc2.png", paths)
    figures.append(FigureItem(
        chapter="Chapter3",
        filename="ch3_pca_scatter_pc1_pc2.png",
        caption="پراکندگی نمونه ها در فضای PC1 و PC2.",
        label="fig:ch3_pca_scatter",
        explanation="این نمودار میزان جداسازی کلاس ها را در فضای PCA نمایش می دهد و شهودی از دشواری طبقه بندی ارائه می کند.",
    ))


def ch3_model_ranking(paths: Paths, figures: List[FigureItem]) -> None:
    summary = _safe_read_excel(paths.results_no_pca, sheet_name="Summary")
    if summary is None or not {"Model", "F1"}.issubset(summary.columns):
        return

    ranking = summary.sort_values("F1", ascending=False).reset_index(drop=True)
    ranking["Rank"] = ranking.index + 1

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=ranking, x="Model", y="F1", palette="mako", ax=ax)
    ax.set_title("Model Ranking by F1 (No PCA)")
    ax.set_ylabel("F1 (%)")
    ax.set_ylim(0, 100)

    _save_dual(fig, "Chapter3", "ch3_model_ranking_f1_no_pca.png", paths)
    figures.append(FigureItem(
        chapter="Chapter3",
        filename="ch3_model_ranking_f1_no_pca.png",
        caption="رتبه بندی مدل ها بر اساس F1 در حالت بدون PCA.",
        label="fig:ch3_model_ranking",
        explanation="این شکل جمع بندی سریع از برتری نسبی مدل ها ارائه می دهد و انتخاب مدل نهایی را پشتیبانی می کند.",
    ))


def ch4_rf_feature_importance(paths: Paths, figures: List[FigureItem], top_n: int = 20) -> None:
    _, X, y, feature_cols = load_master_xy(paths.master_features)
    if X is None or y is None or not feature_cols:
        return

    rf = RandomForestClassifier(n_estimators=500, random_state=42)
    rf.fit(X, y)

    importances = pd.DataFrame({"Feature": feature_cols, "Importance": rf.feature_importances_})
    importances = importances.sort_values("Importance", ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=(10, 7))
    sns.barplot(data=importances, x="Importance", y="Feature", ax=ax, palette="viridis")
    ax.set_title(f"Top {top_n} Feature Importances (Random Forest)")

    _save_dual(fig, "Chapter4", "ch4_rf_top_feature_importance.png", paths)
    figures.append(FigureItem(
        chapter="Chapter4",
        filename="ch4_rf_top_feature_importance.png",
        caption="20 ویژگی برتر بر اساس اهمیت در مدل Random Forest.",
        label="fig:ch4_rf_importance",
        explanation="این شکل نشان می دهد کدام ویژگی ها نقش بیشتری در تصمیم گیری مدل داشته اند و برای تفسیر بالینی/مهندسی مفید است.",
    ))


def ch4_per_class_recall(paths: Paths, figures: List[FigureItem], use_pca: bool) -> None:
    _, X, y, _ = load_master_xy(paths.master_features)
    if X is None or y is None:
        return

    models = _get_models(use_pca=use_pca)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    labels = np.unique(y)

    rows = []
    for model_name, model in models.items():
        y_pred = cross_val_predict(model, X, y, cv=cv)
        cm = confusion_matrix(y, y_pred, labels=labels, normalize="true")
        for i, label in enumerate(labels):
            rows.append({"Model": model_name, "Label": str(label), "Recall": cm[i, i] * 100})

    recall_df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=recall_df, x="Label", y="Recall", hue="Model", ax=ax, palette="Set1")
    ax.set_ylim(0, 100)
    setup_name = "pca95" if use_pca else "no_pca"
    ax.set_title(f"Per-Class Recall Comparison ({setup_name})")

    fname = f"ch4_per_class_recall_{setup_name}.png"
    _save_dual(fig, "Chapter4", fname, paths)
    figures.append(FigureItem(
        chapter="Chapter4",
        filename=fname,
        caption=f"مقایسه Recall هر کلاس بین مدل ها ({setup_name}).",
        label=f"fig:ch4_per_class_recall_{setup_name}",
        explanation="این نمودار حساسیت هر مدل نسبت به هر کلاس را نشان می دهد و برای تحلیل خطاهای کلاس محور استفاده می شود.",
    ))


def ch4_pca_gain(paths: Paths, figures: List[FigureItem]) -> None:
    no_pca = _safe_read_excel(paths.results_no_pca, sheet_name="Summary")
    pca = _safe_read_excel(paths.results_pca95, sheet_name="Summary")
    if no_pca is None or pca is None:
        return

    need_cols = {"Model", "F1"}
    if not need_cols.issubset(no_pca.columns) or not need_cols.issubset(pca.columns):
        return

    merged = no_pca[["Model", "F1"]].merge(
        pca[["Model", "F1"]], on="Model", suffixes=("_NoPCA", "_PCA95"), how="inner"
    )
    if merged.empty:
        return

    merged["F1_Gain"] = merged["F1_PCA95"] - merged["F1_NoPCA"]

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=merged, x="Model", y="F1_Gain", palette="coolwarm", ax=ax)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_title("F1 Gain from PCA95 vs No PCA")
    ax.set_ylabel("F1 Difference (percentage points)")

    _save_dual(fig, "Chapter4", "ch4_f1_gain_pca_vs_no_pca.png", paths)
    figures.append(FigureItem(
        chapter="Chapter4",
        filename="ch4_f1_gain_pca_vs_no_pca.png",
        caption="تغییر F1 هر مدل پس از اعمال PCA95.",
        label="fig:ch4_f1_gain_pca",
        explanation="این شکل مشخص می کند کاهش بعد برای هر مدل سودمند بوده یا باعث افت عملکرد شده است.",
    ))


def ch4_class_centroid_heatmap(paths: Paths, figures: List[FigureItem]) -> None:
    df, _, _, feature_cols = load_master_xy(paths.master_features)
    if df is None or not feature_cols:
        return

    num_df = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    num_df = num_df.fillna(num_df.median(numeric_only=True))
    var_series = num_df.var(numeric_only=True)
    if not isinstance(var_series, pd.Series):
        return
    top_features = var_series.sort_values(ascending=False).head(15).index.tolist()
    if not top_features:
        return

    z = (num_df[top_features] - num_df[top_features].mean()) / (num_df[top_features].std(ddof=0) + 1e-9)
    class_z = z.groupby(df["label"]).mean()

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(class_z, cmap="vlag", center=0, ax=ax)
    ax.set_title("Class-wise Mean Z-Score of Top Features")
    ax.set_xlabel("Feature")
    ax.set_ylabel("Label")

    _save_dual(fig, "Chapter4", "ch4_class_centroid_feature_heatmap.png", paths)
    figures.append(FigureItem(
        chapter="Chapter4",
        filename="ch4_class_centroid_feature_heatmap.png",
        caption="میانگین نرمال شده ویژگی های برتر برای هر کلاس.",
        label="fig:ch4_class_centroid_heatmap",
        explanation="این نقشه حرارتی الگوی میانگین ویژگی ها در کلاس های مختلف را نشان می دهد و قابلیت تفکیک ویژگی ها را تفسیرپذیر می کند.",
    ))


def write_latex_guide(paths: Paths, figures: List[FigureItem]) -> None:
    _mkdir(paths.tex_out.parent)

    lines: List[str] = []
    lines.append("% Auto-generated by Python/Plot/plot_report_figures.py")
    lines.append("% Paste this file with: \\input{Chapters/generated_plot_figures}")
    lines.append("")

    chapter_order = ["Chapter2", "Chapter3", "Chapter4"]
    chapter_title = {
        "Chapter2": "فصل ۲ - نمودارهای داده و پیش پردازش",
        "Chapter3": "فصل ۳ - نمودارهای نتایج مدل ها",
        "Chapter4": "فصل ۴ - نمودارهای تحلیل و جمع بندی",
    }

    for ch in chapter_order:
        subset = [f for f in figures if f.chapter == ch]
        if not subset:
            continue

        lines.append(f"\\subsection*{{{chapter_title[ch]}}}")
        lines.append("")

        for item in subset:
            rel_report_path = f"Images/Generated/{item.chapter}/{item.filename}"
            lines.append("\\begin{figure}[H]")
            lines.append("    \\centering")
            lines.append(f"    \\includegraphics[width=0.85\\linewidth]{{{rel_report_path}}}")
            lines.append(f"    \\caption{{{item.caption}}}")
            lines.append(f"    \\label{{{item.label}}}")
            lines.append("\\end{figure}")
            lines.append("")
            lines.append("\\noindent\\textbf{توضیح شکل:} " + item.explanation)
            lines.append("\\vspace{0.8em}")
            lines.append("")

    paths.tex_out.write_text("\n".join(lines), encoding="utf-8")


def print_outputs(paths: Paths, figures: List[FigureItem]) -> None:
    print("\nGenerated figure files (saved in both folders):")
    for item in figures:
        p1 = paths.report_images_out / item.chapter / item.filename
        p2 = paths.results_images_out / item.chapter / item.filename
        s1 = "OK" if p1.exists() else "MISSING"
        s2 = "OK" if p2.exists() else "MISSING"
        print(f"[{s1}] {p1}")
        print(f"[{s2}] {p2}")

    print("\nLaTeX guide file:")
    print(paths.tex_out)


def build_all(paths: Paths) -> List[FigureItem]:
    figures: List[FigureItem] = []

    ch2_signal_sample(paths, figures)
    ch2_class_distribution(paths, figures)
    ch2_correlation_heatmap(paths, figures)

    ch3_metric_comparison(paths, figures)
    ch3_metric_boxplot(paths, figures)
    ch3_confusion_matrices(paths, figures, use_pca=False)
    ch3_confusion_matrices(paths, figures, use_pca=True)
    ch3_pca_variance(paths, figures)
    ch3_pca_scatter(paths, figures)
    ch3_model_ranking(paths, figures)

    ch4_rf_feature_importance(paths, figures)
    ch4_per_class_recall(paths, figures, use_pca=False)
    ch4_per_class_recall(paths, figures, use_pca=True)
    ch4_pca_gain(paths, figures)
    ch4_class_centroid_heatmap(paths, figures)

    write_latex_guide(paths, figures)
    return figures


if __name__ == "__main__":
    this_file = Path(__file__).resolve()
    project_root = this_file.parents[2]
    paths = infer_paths(project_root)

    print("Project root:", paths.project_root)
    print("Report output:", paths.report_images_out)
    print("Results output:", paths.results_images_out)

    figures_done = build_all(paths)
    print_outputs(paths, figures_done)

    print("\nDone.")
