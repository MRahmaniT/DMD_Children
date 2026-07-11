import os
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# =====================================================
# SETTINGS
# =====================================================

# 1. Choose task: Stand / Sit_To_Stand / Jump / ...
TASK = "Jump"

# 2. Enable/disable action-detected scenarios.
# Baseline (non-detected data) is always executed.
DETECTED = False

# 3. Enable/disable PCA scenario families.
# Baseline without PCA is always executed.
# Pipeline PCA and Fixed PCA are treated as separate alternatives;
# they are never applied together in one scenario.
USE_PIPELINE_PCA = True
USE_FIXED_PCA = False

# 4. PCA retained variance
PCA_VARIANCE = 0.95

# 5. Repeated stratified K-fold settings
N_SPLITS = 5
REPEATS = 5
RANDOM_SEED = 42

# 6. Model parameters
RF_TREES = 500
KNN_NEIGHBORS = 5

# 7. Project paths
PROJECT_ROOT = Path(
    "/Users/mohammad/University/Bachelor Project/Final"
)
RESULTS_ROOT = Path("Results") / TASK

# Metrics shown in Excel and plots
METRICS = ["Accuracy", "Precision", "Recall", "Specificity", "F1"]
MODEL_ORDER = ["KNN", "SVM", "Random Forest"]


# =====================================================
# SCENARIO GENERATION
# =====================================================

def build_scenarios() -> list[dict[str, Any]]:
    """
    Build all allowed scenarios from the enable flags.

    Examples
    --------
    DETECTED=True, USE_PIPELINE_PCA=True, USE_FIXED_PCA=False:
        1. Baseline
        2. Pipeline_PCA
        3. Detected
        4. Detected_Pipeline_PCA

    If all three flags are True, six valid scenarios are run:
        normal/detected x no-PCA/pipeline-PCA/fixed-PCA.
    Pipeline PCA and Fixed PCA are never combined with each other.
    """
    detection_modes = [False, True] if DETECTED else [False]

    pca_modes = ["none"]
    if USE_PIPELINE_PCA:
        pca_modes.append("pipeline")
    if USE_FIXED_PCA:
        pca_modes.append("fixed")

    scenarios: list[dict[str, Any]] = []

    for detected_mode in detection_modes:
        for pca_mode in pca_modes:
            if not detected_mode and pca_mode == "none":
                scenario_name = "Baseline"
            elif detected_mode and pca_mode == "none":
                scenario_name = "Detected_Action"
            elif not detected_mode and pca_mode == "pipeline":
                scenario_name = "Pipeline_PCA"
            elif detected_mode and pca_mode == "pipeline":
                scenario_name = "Detected_Action_Pipeline_PCA"
            elif not detected_mode and pca_mode == "fixed":
                scenario_name = "Fixed_PCA"
            elif detected_mode and pca_mode == "fixed":
                scenario_name = "Detected_Action_Fixed_PCA"
            else:
                raise ValueError("Unsupported scenario configuration.")

            scenarios.append(
                {
                    "name": scenario_name,
                    "detected": detected_mode,
                    "pca_mode": pca_mode,
                }
            )

    # Keep the baseline first, then simpler scenarios before combined ones.
    order = {
        "Baseline": 0,
        "Pipeline_PCA": 1,
        "Fixed_PCA": 2,
        "Detected_Action": 3,
        "Detected_Action_Pipeline_PCA": 4,
        "Detected_Action_Fixed_PCA": 5,
    }
    scenarios.sort(key=lambda item: order[item["name"]])
    return scenarios


def get_scenario_paths(scenario: dict[str, Any]) -> tuple[Path, Path, Path]:
    """Return input Excel, scenario folder, and output Excel paths."""
    detected_mode = bool(scenario["detected"])
    pca_mode = str(scenario["pca_mode"])
    scenario_name = str(scenario["name"])

    if detected_mode:
        data_root = PROJECT_ROOT / "DetectedActionData" / TASK / "Master Features"
    else:
        data_root = PROJECT_ROOT / "Data" / TASK / "Master Features"

    if pca_mode == "fixed":
        master_filename = (
            f"MASTER_Features_{TASK}_PCA{int(PCA_VARIANCE * 100)}.xlsx"
        )
    else:
        master_filename = f"MASTER_Features_{TASK}.xlsx"

    master_path = data_root / master_filename
    scenario_folder = RESULTS_ROOT / scenario_name
    output_excel = scenario_folder / f"Results_{scenario_name}_{TASK}.xlsx"

    return master_path, scenario_folder, output_excel


# =====================================================
# DATA LOADING
# =====================================================

def load_features(master_path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not master_path.exists():
        raise FileNotFoundError(f"Input dataset was not found:\n{master_path}")

    df = pd.read_excel(master_path)
    df = df.drop(
        columns=[c for c in ["source_file", "prefix", "idx"] if c in df.columns],
        errors="ignore",
    )

    if "label" not in df.columns:
        raise KeyError(f"The 'label' column is missing from:\n{master_path}")

    y = df["label"].astype(int).to_numpy()
    X = df.drop(columns=["label"]).copy()

    for column in X.columns:
        X[column] = pd.to_numeric(X[column], errors="coerce")

    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True))

    # Columns that are still entirely NaN are replaced by zero.
    X = X.fillna(0)

    return X.to_numpy(dtype=float), y


# =====================================================
# METRICS
# =====================================================

def multiclass_specificity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    labels = np.unique(np.concatenate([y_true, y_pred]))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    specificities: list[float] = []

    for i in range(cm.shape[0]):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        tn = cm.sum() - tp - fp - fn
        denominator = tn + fp
        specificities.append(tn / denominator if denominator > 0 else 0.0)

    return float(np.mean(specificities))


# =====================================================
# MODELS
# =====================================================

def get_models(
    seed: int,
    pca_mode: str,
    pca_variance: float,
) -> dict[str, Pipeline]:
    if pca_mode == "pipeline":
        knn_steps = [
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=pca_variance, random_state=seed)),
            ("clf", KNeighborsClassifier(n_neighbors=KNN_NEIGHBORS)),
        ]
        svm_steps = [
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=pca_variance, random_state=seed)),
            ("clf", SVC(kernel="rbf")),
        ]
        rf_steps = [
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=pca_variance, random_state=seed)),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=RF_TREES,
                    random_state=seed,
                ),
            ),
        ]
    elif pca_mode == "fixed":
        # The fixed-PCA Excel file is assumed to already contain PCA scores.
        knn_steps = [
            ("clf", KNeighborsClassifier(n_neighbors=KNN_NEIGHBORS)),
        ]
        svm_steps = [
            ("clf", SVC(kernel="rbf")),
        ]
        rf_steps = [
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=RF_TREES,
                    random_state=seed,
                ),
            ),
        ]
    elif pca_mode == "none":
        knn_steps = [
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier(n_neighbors=KNN_NEIGHBORS)),
        ]
        svm_steps = [
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf")),
        ]
        rf_steps = [
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=RF_TREES,
                    random_state=seed,
                ),
            ),
        ]
    else:
        raise ValueError(f"Unknown PCA mode: {pca_mode}")

    return {
        "KNN": Pipeline(knn_steps),
        "SVM": Pipeline(svm_steps),
        "Random Forest": Pipeline(rf_steps),
    }


# =====================================================
# PLOTTING
# =====================================================

def save_scenario_plots(
    summary_df: pd.DataFrame,
    all_runs_df: pd.DataFrame,
    scenario_name: str,
    scenario_folder: Path,
) -> None:
    """
    Save two plots for each scenario:

    1. Grouped boxplots for Accuracy, Precision, Recall, Specificity, and F1
       across all repeated folds, with one box per model.
    2. A separate grouped bar chart showing mean Accuracy for each model.
    """
    scenario_folder.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------
    # Plot 1: grouped boxplots for all five metrics
    # -------------------------------------------------
    n_metrics = len(METRICS)
    n_models = len(MODEL_ORDER)

    group_centers = np.arange(1, n_metrics + 1, dtype=float)
    box_width = 0.22
    offsets = np.linspace(
        -box_width * (n_models - 1) / 2,
        box_width * (n_models - 1) / 2,
        n_models,
    )

    fig, ax = plt.subplots(figsize=(11, 6))

    legend_handles = []

    for model_index, model_name in enumerate(MODEL_ORDER):
        model_data = []
        for metric in METRICS:
            values = all_runs_df.loc[
                all_runs_df["Model"] == model_name, metric
            ].to_numpy(dtype=float)
            model_data.append(values)

        positions = group_centers + offsets[model_index]

        boxplot = ax.boxplot(
            model_data,
            positions=positions,
            widths=box_width * 0.9,
            patch_artist=True,
            manage_ticks=False,
            showfliers=True,
            medianprops={"color": "black", "linewidth": 1.2},
            whiskerprops={"linewidth": 1.0},
            capprops={"linewidth": 1.0},
            flierprops={
                "marker": "o",
                "markersize": 4,
                "markerfacecolor": "white",
                "markeredgecolor": "black",
                "alpha": 0.7,
            },
        )

        # Use matplotlib's default color cycle, without hard-coding colors.
        default_color = plt.rcParams["axes.prop_cycle"].by_key()["color"][
            model_index % len(plt.rcParams["axes.prop_cycle"].by_key()["color"])
        ]

        for box in boxplot["boxes"]:
            box.set_facecolor(default_color)
            box.set_alpha(0.8)

        legend_handles.append(
            Rectangle(
                (0, 0),
                1,
                1,
                facecolor=default_color,
                alpha=0.8,
                label=model_name,
            )
        )

    ax.set_title(f"Metric Distribution Across Folds - {scenario_name}")
    ax.set_xlabel("Metric")
    ax.set_ylabel("Score (%)")
    ax.set_xticks(group_centers)
    ax.set_xticklabels(METRICS)
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(handles=legend_handles, title="Model", loc="lower left")

    fig.tight_layout()
    fig.savefig(
        scenario_folder / f"Metric_Distribution_{scenario_name}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    # -------------------------------------------------
    # Plot 2: mean Accuracy as a bar chart
    # -------------------------------------------------
    accuracy_df = (
        summary_df[["Model", "Accuracy"]]
        .set_index("Model")
        .reindex(MODEL_ORDER)
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(8, 5.5))
    bars = ax.bar(
        accuracy_df["Model"],
        accuracy_df["Accuracy"],
    )

    ax.set_title(f"Mean Accuracy Comparison - {scenario_name}")
    ax.set_xlabel("Model")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.3)

    for bar, value in zip(bars, accuracy_df["Accuracy"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            float(value) + 1.0,
            f"{float(value):.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.tight_layout()
    fig.savefig(
        scenario_folder / f"Accuracy_Bar_{scenario_name}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def save_pairwise_accuracy_plot(
    baseline_summary: pd.DataFrame,
    scenario_summary: pd.DataFrame,
    scenario_name: str,
    comparison_folder: Path,
) -> pd.DataFrame:
    baseline = baseline_summary[["Model", "Accuracy"]].rename(
        columns={"Accuracy": "Baseline"}
    )
    scenario = scenario_summary[["Model", "Accuracy"]].rename(
        columns={"Accuracy": scenario_name}
    )

    comparison_df = baseline.merge(scenario, on="Model", how="inner")
    comparison_df["Accuracy_Difference"] = (
        comparison_df[scenario_name] - comparison_df["Baseline"]
    )
    comparison_df = comparison_df.set_index("Model").reindex(MODEL_ORDER).reset_index()

    ax = comparison_df.set_index("Model")[["Baseline", scenario_name]].plot(
        kind="bar",
        figsize=(10, 6),
    )
    ax.set_title(f"Accuracy comparison: Baseline vs {scenario_name}")
    ax.set_xlabel("Model")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(
        comparison_folder / f"Accuracy_Baseline_vs_{scenario_name}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    return comparison_df


def save_overall_accuracy_plot(
    scenario_results: dict[str, dict[str, pd.DataFrame]],
    comparison_folder: Path,
) -> None:
    rows: list[pd.DataFrame] = []

    for scenario_name, result in scenario_results.items():
        temp = result["summary"][["Model", "Accuracy"]].copy()
        temp["Scenario"] = scenario_name
        rows.append(temp)

    combined = pd.concat(rows, ignore_index=True)
    pivot = combined.pivot(index="Model", columns="Scenario", values="Accuracy")
    pivot = pivot.reindex(MODEL_ORDER)

    ax = pivot.plot(kind="bar", figsize=(13, 7))
    ax.set_title("Accuracy comparison across all executed scenarios")
    ax.set_xlabel("Model")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(
        comparison_folder / "Overall_Accuracy_Comparison.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


# =====================================================
# ONE-SCENARIO EVALUATION
# =====================================================

def run_one_scenario(
    scenario: dict[str, Any],
) -> dict[str, pd.DataFrame]:
    scenario_name = str(scenario["name"])
    pca_mode = str(scenario["pca_mode"])
    detected_mode = bool(scenario["detected"])

    master_path, scenario_folder, output_excel = get_scenario_paths(scenario)
    scenario_folder.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print(f"SCENARIO: {scenario_name}")
    print("=" * 80)
    print(f"Detected data: {detected_mode}")
    print(f"PCA mode: {pca_mode}")
    print(f"Input: {master_path}")

    X, y = load_features(master_path)

    print(f"Dataset shape: {X.shape}")
    print(f"Classes: {np.unique(y)}")

    splitter = RepeatedStratifiedKFold(
        n_splits=N_SPLITS,
        n_repeats=REPEATS,
        random_state=RANDOM_SEED,
    )

    all_runs: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    for model_name in MODEL_ORDER:
        metric_values = {metric: [] for metric in METRICS}
        print(f"Running {model_name}...")

        for fold_number, (train_indices, test_indices) in enumerate(
            splitter.split(X, y),
            start=1,
        ):
            X_train, y_train = X[train_indices], y[train_indices]
            X_test, y_test = X[test_indices], y[test_indices]

            model = get_models(
                seed=RANDOM_SEED,
                pca_mode=pca_mode,
                pca_variance=PCA_VARIANCE,
            )[model_name]

            model.fit(X_train, y_train)
            predictions: np.ndarray = np.asarray(model.predict(X_test))

            fold_metrics = {
                "Accuracy": accuracy_score(y_test, predictions) * 100,
                "Precision": precision_score(
                    y_test,
                    predictions,
                    average="macro",
                    zero_division=0,
                )
                * 100,
                "Recall": recall_score(
                    y_test,
                    predictions,
                    average="macro",
                    zero_division=0,
                )
                * 100,
                "Specificity": multiclass_specificity(y_test, predictions) * 100,
                "F1": f1_score(
                    y_test,
                    predictions,
                    average="macro",
                    zero_division=0,
                )
                * 100,
            }

            for metric_name, metric_value in fold_metrics.items():
                metric_values[metric_name].append(metric_value)

            all_runs.append(
                {
                    "Scenario": scenario_name,
                    "Model": model_name,
                    "Fold": fold_number,
                    **fold_metrics,
                }
            )

        summary_row: dict[str, Any] = {
            "Scenario": scenario_name,
            "Model": model_name,
        }
        for metric_name in METRICS:
            values = np.asarray(metric_values[metric_name], dtype=float)
            summary_row[metric_name] = float(np.mean(values))
            summary_row[f"{metric_name}_Std"] = float(np.std(values, ddof=1))

        summary_rows.append(summary_row)

    summary_df = pd.DataFrame(summary_rows).sort_values("F1", ascending=False)
    ranking_df = summary_df.copy()
    ranking_df.insert(0, "Rank", range(1, len(ranking_df) + 1))
    all_runs_df = pd.DataFrame(all_runs)

    settings_df = pd.DataFrame(
        [
            {
                "Task": TASK,
                "Scenario": scenario_name,
                "Detected": detected_mode,
                "PCA_Mode": pca_mode,
                "PCA_Variance": PCA_VARIANCE if pca_mode != "none" else "Not used",
                "N_Splits": N_SPLITS,
                "Repeats": REPEATS,
                "Random_Seed": RANDOM_SEED,
                "RF_Trees": RF_TREES,
                "KNN_Neighbors": KNN_NEIGHBORS,
                "Input_File": str(master_path),
            }
        ]
    )

    with pd.ExcelWriter(output_excel, engine="openpyxl") as writer:
        settings_df.to_excel(writer, sheet_name="Settings", index=False)
        summary_df.to_excel(writer, sheet_name="Summary", index=False)
        ranking_df.to_excel(writer, sheet_name="Ranking", index=False)
        all_runs_df.to_excel(writer, sheet_name="All_Folds", index=False)

    save_scenario_plots(
        summary_df=summary_df,
        all_runs_df=all_runs_df,
        scenario_name=scenario_name,
        scenario_folder=scenario_folder,
    )

    print(summary_df.round(3).to_string(index=False))
    print(f"Saved scenario output: {output_excel.resolve()}")

    return {
        "summary": summary_df,
        "ranking": ranking_df,
        "all_runs": all_runs_df,
        "settings": settings_df,
    }


# =====================================================
# OVERALL COMPARISON
# =====================================================

def save_overall_comparisons(
    scenario_results: dict[str, dict[str, pd.DataFrame]],
) -> None:
    comparison_folder = RESULTS_ROOT / "Comparisons"
    comparison_folder.mkdir(parents=True, exist_ok=True)

    baseline_summary = scenario_results["Baseline"]["summary"]

    all_summaries: list[pd.DataFrame] = []
    pairwise_tables: dict[str, pd.DataFrame] = {}

    for scenario_name, result in scenario_results.items():
        all_summaries.append(result["summary"].copy())

        if scenario_name != "Baseline":
            pairwise_tables[scenario_name] = save_pairwise_accuracy_plot(
                baseline_summary=baseline_summary,
                scenario_summary=result["summary"],
                scenario_name=scenario_name,
                comparison_folder=comparison_folder,
            )

    combined_summary_df = pd.concat(all_summaries, ignore_index=True)

    # A long-format table containing each scenario's difference from baseline
    # for every model and every metric.
    baseline_metrics = baseline_summary[["Model", *METRICS]].copy()
    baseline_metrics = baseline_metrics.rename(
        columns={metric: f"Baseline_{metric}" for metric in METRICS}
    )

    delta_rows: list[pd.DataFrame] = []
    for scenario_name, result in scenario_results.items():
        if scenario_name == "Baseline":
            continue

        current = result["summary"][["Model", *METRICS]].copy()
        merged = current.merge(baseline_metrics, on="Model", how="inner")
        merged.insert(0, "Scenario", scenario_name)

        for metric in METRICS:
            merged[f"Delta_{metric}"] = (
                merged[metric] - merged[f"Baseline_{metric}"]
            )

        delta_rows.append(merged)

    if delta_rows:
        deltas_df = pd.concat(delta_rows, ignore_index=True)
    else:
        deltas_df = pd.DataFrame()

    comparison_excel = comparison_folder / f"Overall_Comparison_{TASK}.xlsx"
    with pd.ExcelWriter(comparison_excel, engine="openpyxl") as writer:
        combined_summary_df.to_excel(
            writer,
            sheet_name="All_Scenarios",
            index=False,
        )

        if not deltas_df.empty:
            deltas_df.to_excel(
                writer,
                sheet_name="Deltas_vs_Baseline",
                index=False,
            )

        for scenario_name, table in pairwise_tables.items():
            # Excel sheet names cannot exceed 31 characters.
            sheet_name = f"Base_vs_{scenario_name}"[:31]
            table.to_excel(writer, sheet_name=sheet_name, index=False)

    save_overall_accuracy_plot(
        scenario_results=scenario_results,
        comparison_folder=comparison_folder,
    )

    print("\n" + "=" * 80)
    print("OVERALL COMPARISONS SAVED")
    print("=" * 80)
    print(comparison_excel.resolve())


# =====================================================
# MAIN
# =====================================================

def main() -> None:
    scenarios = build_scenarios()
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)

    print("Scenarios to execute:")
    for number, scenario in enumerate(scenarios, start=1):
        print(f"{number}. {scenario['name']}")

    scenario_results: dict[str, dict[str, pd.DataFrame]] = {}
    failed_scenarios: list[dict[str, str]] = []

    for scenario in scenarios:
        scenario_name = str(scenario["name"])
        try:
            scenario_results[scenario_name] = run_one_scenario(scenario)
        except Exception as exc:
            failed_scenarios.append(
                {
                    "Scenario": scenario_name,
                    "Error": str(exc),
                }
            )
            print(f"\nERROR in {scenario_name}: {exc}")

    if "Baseline" not in scenario_results:
        raise RuntimeError(
            "Baseline failed, so comparison files cannot be generated. "
            "Check the baseline input path and dataset."
        )

    save_overall_comparisons(scenario_results)

    if failed_scenarios:
        failed_df = pd.DataFrame(failed_scenarios)
        failed_path = RESULTS_ROOT / "Comparisons" / "Failed_Scenarios.xlsx"
        failed_df.to_excel(failed_path, index=False)
        print("\nSome scenarios failed. Details were saved to:")
        print(failed_path.resolve())

    print("\nFinished.")
    print(f"Main output folder: {RESULTS_ROOT.resolve()}")


if __name__ == "__main__":
    main()
