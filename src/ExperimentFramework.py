"""Shared, method-agnostic framework for report-grade experiments.

The normal lab entry point trains one set of models on one split.  This module
adds the parts needed by report Sections 2-5: repeated target-aware splits,
immutable study folders, uncertainty summaries and a small set of
decision-relevant artifacts.
"""

from __future__ import annotations

import csv
import json
import platform
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import matplotlib
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PRIMARY_METRICS = (
    "voltage_V_mae",
    "voltage_V_rmse",
    "voltage_V_r2",
    "position_cm_mae",
    "position_cm_rmse",
    "position_cm_r2",
    "mean_normalised_rmse",
    "generalisation_gap_mean_normalised_rmse",
    "parameters",
    "epochs_run",
    "training_seconds",
)


@dataclass(frozen=True)
class ExperimentSpec:
    """One model/representation choice declared in the study JSON."""

    experiment_id: str
    method: str
    model_kind: str
    architecture: str
    noise_standard_deviation: float
    feature_set: str
    feature_window: str
    fundamental_rule: str
    collect_xai: bool


def loadStudyConfiguration(configurationFile):
    """Load and validate the stable JSON interface used by future studies."""
    configurationPath = Path(configurationFile).resolve()
    with configurationPath.open(encoding="utf-8") as file:
        configuration = json.load(file)

    requiredTopLevel = ("study_name", "splits", "training", "experiments")
    missing = [name for name in requiredTopLevel if name not in configuration]
    if missing:
        raise ValueError(f"Study configuration is missing: {missing}")

    experiments = []
    seenIdentifiers = set()
    for rawExperiment in configuration["experiments"]:
        experiment = ExperimentSpec(
            experiment_id=str(rawExperiment["id"]),
            method=str(rawExperiment["method"]),
            model_kind=str(rawExperiment.get("model_kind", "neural")),
            architecture=str(rawExperiment.get("architecture", "adopted")),
            noise_standard_deviation=float(
                rawExperiment.get("noise_standard_deviation", 0.0)
            ),
            feature_set=str(rawExperiment.get("feature_set", "all")),
            feature_window=str(
                rawExperiment.get("feature_window", "rectangular")
            ),
            fundamental_rule=str(
                rawExperiment.get("fundamental_rule", "band_peak")
            ),
            collect_xai=bool(rawExperiment.get("collect_xai", False)),
        )
        if experiment.experiment_id in seenIdentifiers:
            raise ValueError(
                f"Duplicate experiment id: {experiment.experiment_id}"
            )
        if experiment.method not in ("feature", "time", "spectrum"):
            raise ValueError(
                f"Unsupported method for {experiment.experiment_id}: "
                f"{experiment.method}"
            )
        if experiment.model_kind not in ("neural", "linear", "ridge"):
            raise ValueError(
                f"Unsupported model_kind for {experiment.experiment_id}: "
                f"{experiment.model_kind}"
            )
        if (
            experiment.model_kind in ("linear", "ridge")
            and experiment.method != "feature"
        ):
            raise ValueError(
                "Linear and Ridge baselines currently require feature input."
            )
        if experiment.noise_standard_deviation < 0:
            raise ValueError("Noise standard deviation cannot be negative.")
        if experiment.feature_window not in ("rectangular", "hann"):
            raise ValueError(
                f"Unsupported feature_window: {experiment.feature_window}"
            )
        if experiment.fundamental_rule not in (
            "band_peak",
            "harmonic_check",
        ):
            raise ValueError(
                "fundamental_rule must be band_peak or harmonic_check."
            )
        seenIdentifiers.add(experiment.experiment_id)
        experiments.append(experiment)

    seeds = [int(seed) for seed in configuration["splits"]["seeds"]]
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Split seeds must be a non-empty unique list.")

    validationFraction = float(
        configuration["splits"].get("validation_fraction", 0.2)
    )
    if not 0.1 <= validationFraction <= 0.5:
        raise ValueError("validation_fraction must be between 0.1 and 0.5.")

    return configurationPath, configuration, experiments


def makeTargetStrata(targets, maximumBins=4):
    """Create robust joint voltage/position strata from target quantiles."""
    targets = np.asarray(targets, dtype=np.float64)
    if targets.ndim != 2 or targets.shape[1] != 2:
        raise ValueError("Expected targets with shape (samples, 2).")

    for numberOfBins in range(int(maximumBins), 1, -1):
        columns = []
        for targetNumber in range(2):
            edges = np.unique(
                np.quantile(
                    targets[:, targetNumber],
                    np.linspace(0.0, 1.0, numberOfBins + 1),
                )
            )
            columns.append(
                np.digitize(targets[:, targetNumber], edges[1:-1])
            )
        labels = columns[0] * numberOfBins + columns[1]
        _, counts = np.unique(labels, return_counts=True)
        if counts.min() >= 2:
            return labels.astype(int), numberOfBins

    # A one-dimensional voltage stratification is still more target-aware
    # than an unconstrained permutation if the joint grid is too sparse.
    edges = np.unique(
        np.quantile(targets[:, 0], np.linspace(0.0, 1.0, 5))
    )
    labels = np.digitize(targets[:, 0], edges[1:-1])
    return labels.astype(int), 4


def makeRepeatedTargetAwareSplits(
    targets,
    seeds,
    validationFraction=0.2,
    maximumBins=4,
):
    """Return one reproducible stratified holdout for every declared seed."""
    strata, binsUsed = makeTargetStrata(targets, maximumBins)
    allIndices = np.arange(len(targets))
    splits = []

    for repeatNumber, seed in enumerate(seeds):
        splitter = StratifiedShuffleSplit(
            n_splits=1,
            test_size=validationFraction,
            random_state=int(seed),
        )
        trainIndices, validationIndices = next(
            splitter.split(allIndices, strata)
        )
        splits.append(
            {
                "repeat": repeatNumber,
                "seed": int(seed),
                "train_indices": trainIndices,
                "validation_indices": validationIndices,
            }
        )
    return splits, strata, binsUsed


def createImmutableStudyFolder(outputRoot, studyName):
    """Create a timestamped folder and refuse any accidental overwrite."""
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    safeName = "".join(
        character if character.isalnum() or character in "-_" else "-"
        for character in studyName.strip()
    ).strip("-")
    studyFolder = Path(outputRoot).resolve() / f"{timestamp}_{safeName}"
    studyFolder.mkdir(parents=True, exist_ok=False)
    return studyFolder


def calculateMetricSummary(metricRows):
    """Aggregate each experiment/metric using course-friendly descriptive stats."""
    groupedValues = {}
    for row in metricRows:
        experimentId = row["experiment_id"]
        for metricName in PRIMARY_METRICS:
            groupedValues.setdefault(
                (experimentId, metricName), []
            ).append(float(row[metricName]))

    summaryRows = []
    for (experimentId, metricName), values in sorted(groupedValues.items()):
        values = np.asarray(values, dtype=np.float64)
        sampleCount = len(values)
        standardDeviation = (
            float(np.std(values, ddof=1)) if sampleCount > 1 else 0.0
        )
        summaryRows.append(
            {
                "experiment_id": experimentId,
                "metric": metricName,
                "repeats": sampleCount,
                "mean": float(np.mean(values)),
                "standard_deviation": standardDeviation,
                "minimum": float(np.min(values)),
                "maximum": float(np.max(values)),
            }
        )
    return summaryRows


def writeRows(filePath, rows):
    """Write a list of dictionaries as one report-ready CSV."""
    rows = list(rows)
    if not rows:
        return
    with Path(filePath).open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def saveStudyArtifacts(
    studyFolder,
    configurationPath,
    configuration,
    experiments,
    splits,
    strata,
    binsUsed,
    metricRows,
    predictionRows,
    packageVersions,
):
    """Save only artifacts that support selection, reporting or reproduction."""
    studyFolder = Path(studyFolder)
    summaryRows = calculateMetricSummary(metricRows)

    writeRows(studyFolder / "run_metrics.csv", metricRows)
    writeRows(studyFolder / "summary_metrics.csv", summaryRows)
    writeRows(studyFolder / "validation_predictions.csv", predictionRows)

    splitRows = []
    for split in splits:
        trainSet = set(int(index) for index in split["train_indices"])
        for sampleIndex in range(len(strata)):
            splitRows.append(
                {
                    "repeat": split["repeat"],
                    "seed": split["seed"],
                    "sample_index": sampleIndex,
                    "target_stratum": int(strata[sampleIndex]),
                    "role": (
                        "training" if sampleIndex in trainSet else "validation"
                    ),
                }
            )
    writeRows(studyFolder / "split_assignments.csv", splitRows)

    artifactNames = [
        "run_metrics.csv",
        "summary_metrics.csv",
        "validation_predictions.csv",
        "split_assignments.csv",
        "performance_summary.png",
        "residual_summary.png",
    ]
    for optionalName in (
        "feature_target_correlations.csv",
        "feature_redundancy.csv",
        "fundamental_rule_changes.csv",
        "feature_shap_repeated.csv",
        "feature_shap_summary.csv",
        "feature_shap_summary.png",
        "time_grouped_relevance_repeated.csv",
        "time_grouped_relevance_summary.csv",
        "time_grouped_relevance_summary.png",
        "spectrum_grouped_relevance_repeated.csv",
        "spectrum_grouped_relevance_summary.csv",
        "spectrum_grouped_relevance_summary.png",
    ):
        if (studyFolder / optionalName).exists():
            artifactNames.append(optionalName)

    manifest = {
        "study_name": configuration["study_name"],
        "status": (
            "complete"
            if len(metricRows) == len(experiments) * len(splits)
            else "in_progress"
        ),
        "completed_runs": len(metricRows),
        "expected_runs": len(experiments) * len(splits),
        "configuration_file": str(configurationPath),
        "created_at_local": datetime.now().isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "packages": packageVersions,
        "target_stratification_bins": binsUsed,
        "experiments": [
            {
                "id": item.experiment_id,
                "method": item.method,
                "model_kind": item.model_kind,
                "architecture": item.architecture,
                "noise_standard_deviation": item.noise_standard_deviation,
                "feature_set": item.feature_set,
                "feature_window": item.feature_window,
                "fundamental_rule": item.fundamental_rule,
                "collect_xai": item.collect_xai,
            }
            for item in experiments
        ],
        "splits": configuration["splits"],
        "training": configuration["training"],
        "artifacts": artifactNames,
        "excluded_by_design": [
            "per-fold Keras models",
            "per-fold training histories",
            "hidden-test predictions during model selection",
        ],
    }
    with (studyFolder / "study_manifest.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(manifest, file, indent=2)

    _savePerformanceFigure(studyFolder, summaryRows)
    _saveResidualFigure(studyFolder, predictionRows)


def _savePerformanceFigure(studyFolder, summaryRows):
    """Save the report anchor plot for repeated-validation performance."""
    experiments = sorted(
        {row["experiment_id"] for row in summaryRows}
    )
    displayLabels = {
        "fcn_hann_adopted_all": "Hann window",
        "fcn_rect_adopted_all": "Rectangular window",
        "fcn_rect_amplitude_shape": "Amplitude features",
        "fcn_rect_frequency_distribution": "Frequency features",
        "fcn_rect_small_all": "Compact network",
        "fcn_rect_wide_all": "Wide network",
        "rect_drop_crest_factor": "No crest factor",
        "rect_drop_dominant_amplitude": "No amplitude",
        "rect_drop_dominant_frequency": "No frequency",
        "rect_drop_high_frequency_ratio": "No high-frequency",
        "rect_drop_kurtosis": "No kurtosis",
        "rect_drop_peak_to_peak": "No peak-to-peak",
        "rect_drop_rms": "No RMS amplitude",
        "rect_drop_spectral_centroid": "No spectral centroid",
        "time_aggressive_control": "Aggressive network",
        "time_gentle_control": "Gentle network",
        "spectrum_average_noise_0": "No noise",
        "spectrum_average_noise_001": "Noise 0.01",
        "spectrum_average_noise_002": "Noise 0.02",
        "spectrum_average_pooling": "Average pooling",
        "spectrum_max_pooling": "Maximum pooling",
    }
    missingLabels = set(experiments) - set(displayLabels)
    if missingLabels:
        raise ValueError(
            "Missing natural-language labels for experiments: "
            f"{sorted(missingLabels)}"
        )
    displayExperiments = [displayLabels[name] for name in experiments]
    metricNames = ("voltage_V_rmse", "position_cm_rmse")
    metricLabels = {
        "voltage_V_rmse": "Voltage RMSE",
        "position_cm_rmse": "Position RMSE",
    }
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    for axis, metricName in zip(axes, metricNames):
        rows = [
            row for row in summaryRows if row["metric"] == metricName
        ]
        byExperiment = {row["experiment_id"]: row for row in rows}
        means = [byExperiment[name]["mean"] for name in experiments]
        errors = [
            byExperiment[name]["standard_deviation"]
            for name in experiments
        ]
        positions = np.arange(len(experiments))
        axis.bar(positions, means, yerr=errors, capsize=4)
        axis.set_xticks(
            positions,
            displayExperiments,
            rotation=35,
            ha="right",
            fontsize=8.5,
        )
        axis.set(
            ylabel="Mean RMSE across splits (SD)",
            title=metricLabels[metricName],
        )
    figure.tight_layout()
    figure.savefig(studyFolder / "performance_summary.png", dpi=200)
    plt.close(figure)


def _saveResidualFigure(studyFolder, predictionRows):
    """Save one compact validation-residual figure for all experiments."""
    experiments = sorted(
        {row["experiment_id"] for row in predictionRows}
    )
    figure, axes = plt.subplots(
        len(experiments),
        2,
        figsize=(9, max(3.5, 3.2 * len(experiments))),
        squeeze=False,
    )

    for rowNumber, experimentId in enumerate(experiments):
        rows = [
            row
            for row in predictionRows
            if row["experiment_id"] == experimentId
        ]
        for targetNumber, (trueName, predictedName, label) in enumerate(
            (
                ("true_voltage_V", "predicted_voltage_V", "Voltage (V)"),
                (
                    "true_position_cm",
                    "predicted_position_cm",
                    "Position (cm)",
                ),
            )
        ):
            trueValues = np.asarray(
                [float(row[trueName]) for row in rows]
            )
            residuals = np.asarray(
                [float(row[predictedName]) for row in rows]
            ) - trueValues
            axes[rowNumber, targetNumber].scatter(
                trueValues, residuals, s=16, alpha=0.55
            )
            axes[rowNumber, targetNumber].axhline(
                0.0, color="black", linestyle="--", linewidth=1
            )
            axes[rowNumber, targetNumber].set(
                xlabel=f"True {label}",
                ylabel="Prediction residual",
                title=f"{experimentId}: {label}",
            )
    figure.tight_layout()
    figure.savefig(studyFolder / "residual_summary.png", dpi=200)
    plt.close(figure)
