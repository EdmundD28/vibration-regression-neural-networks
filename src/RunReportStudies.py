"""Run configuration-driven repeated experiments for report Sections 2-5.

This entry point intentionally saves no per-fold models, histories or hidden
test predictions.  It produces only model-selection, uncertainty, residual and
reproducibility evidence.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import time

import numpy as np
from sklearn.linear_model import LinearRegression, Ridge

from DataAnalysis_Common import (
    DATA_FOLDER,
    FEATURE_NAMES,
    PROJECT_FOLDER,
    calculateGroupedReplacementRelevance,
    calculateRegressionMetrics,
    calculateShapFeatureImportance,
    convertADCToAcceleration,
    fitStandardiser,
    loadData,
    setReproducibleSeed,
    trainAndEvaluateModel,
)
from ExperimentFramework import (
    createImmutableStudyFolder,
    loadStudyConfiguration,
    makeRepeatedTargetAwareSplits,
    saveStudyArtifacts,
    writeRows,
)
from FeatureFCN_Analysis import buildFeatureFCN, calculateFeatures
from SpectrumNetwork_Analysis import (
    buildSpectrumNetwork,
    prepareSpectrumInputs,
)
from TimeSeriesCNN_Analysis import (
    buildTimeSeriesCNN,
    prepareTimeSeriesInputs,
)


FEATURE_SETS = {
    "all": tuple(range(len(FEATURE_NAMES))),
    "amplitude_shape": (0, 1, 2, 3, 5),
    "frequency_distribution": (4, 6, 7),
}


def parseArguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_FOLDER / "experiments" / "report_study.json",
    )
    parser.add_argument("--data-dir", type=Path, default=DATA_FOLDER)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_FOLDER / "outputs" / "studies",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate data, configuration and target-aware splits only.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Use one split and one epoch while testing framework integration.",
    )
    return parser.parse_args()


def selectFeatureColumns(featureSet):
    if featureSet.startswith("drop:"):
        featureName = featureSet.split(":", 1)[1]
        if featureName not in FEATURE_NAMES:
            raise ValueError(f"Unknown feature to drop: {featureName}")
        droppedIndex = FEATURE_NAMES.index(featureName)
        return tuple(
            index
            for index in range(len(FEATURE_NAMES))
            if index != droppedIndex
        )
    if featureSet not in FEATURE_SETS:
        raise ValueError(f"Unknown feature_set: {featureSet}")
    return FEATURE_SETS[featureSet]


def prepareRepresentations(
    xTrain,
    xTest,
    trainIndices,
):
    timeTrain, timeTest = prepareTimeSeriesInputs(
        xTrain, xTest, trainIndices
    )
    spectrumTrain, spectrumTest, _, frequencies = prepareSpectrumInputs(
        xTrain, xTest, trainIndices
    )
    return {
        "time": (timeTrain, timeTest),
        "spectrum": (spectrumTrain, spectrumTest),
        "frequencies": frequencies,
    }


def buildNeuralModel(tf, experiment, inputShape, seed):
    if experiment.method == "feature":
        return buildFeatureFCN(
            tf,
            experiment.noise_standard_deviation,
            modelName=experiment.experiment_id,
            layerSeed=seed,
            numberOfFeatures=inputShape[0],
            architecture=experiment.architecture,
        )
    if experiment.method == "time":
        return buildTimeSeriesCNN(
            tf,
            experiment.noise_standard_deviation,
            modelName=experiment.experiment_id,
            layerSeed=seed,
            architecture=experiment.architecture,
        )
    return buildSpectrumNetwork(
        tf,
        inputShape[0],
        experiment.noise_standard_deviation,
        modelName=experiment.experiment_id,
        layerSeed=seed,
        architecture=experiment.architecture,
    )


def runClassicalBaseline(
    experiment,
    xValues,
    yTrainNorm,
    yTrain,
    yMean,
    yStd,
    trainIndices,
    validationIndices,
):
    model = (
        LinearRegression()
        if experiment.model_kind == "linear"
        else Ridge(alpha=1.0)
    )
    startTime = time.perf_counter()
    model.fit(xValues[trainIndices], yTrainNorm[trainIndices])
    trainingSeconds = time.perf_counter() - startTime
    trainingPrediction = (
        model.predict(xValues[trainIndices]) * yStd + yMean
    )
    validationPrediction = (
        model.predict(xValues[validationIndices]) * yStd + yMean
    )
    metrics = calculateRegressionMetrics(
        yTrain[validationIndices], validationPrediction, yStd
    )
    trainingMetrics = calculateRegressionMetrics(
        yTrain[trainIndices], trainingPrediction, yStd
    )
    metrics.update(
        {
            f"training_{name}": value
            for name, value in trainingMetrics.items()
        }
    )
    metrics["generalisation_gap_mean_normalised_rmse"] = (
        metrics["mean_normalised_rmse"]
        - metrics["training_mean_normalised_rmse"]
    )
    metrics.update(
        {
            "parameters": int(
                np.size(model.coef_) + np.size(model.intercept_)
            ),
            "epochs_run": 0,
            "training_seconds": float(trainingSeconds),
        }
    )
    return metrics, validationPrediction


def saveFeatureDiagnostics(studyFolder, xTrain, yTrain, experiments):
    """Save compact Section 2 correlation, redundancy and rule evidence."""
    configurations = sorted(
        {
            (item.feature_window, item.fundamental_rule)
            for item in experiments
            if item.method == "feature"
        }
    )
    correlationRows = []
    redundancyRows = []
    featureCache = {}

    for windowName, fundamentalRule in configurations:
        values = calculateFeatures(
            xTrain,
            windowName=windowName,
            fundamentalRule=fundamentalRule,
        )
        featureCache[(windowName, fundamentalRule)] = values
        for featureNumber, featureName in enumerate(FEATURE_NAMES):
            for targetNumber, targetName in enumerate(
                ("voltage_V", "position_cm")
            ):
                correlationRows.append(
                    {
                        "feature_window": windowName,
                        "fundamental_rule": fundamentalRule,
                        "feature": featureName,
                        "target": targetName,
                        "pearson_r": float(
                            np.corrcoef(
                                values[:, featureNumber],
                                yTrain[:, targetNumber],
                            )[0, 1]
                        ),
                    }
                )
        for firstIndex, firstName in enumerate(FEATURE_NAMES):
            for secondIndex in range(firstIndex + 1, len(FEATURE_NAMES)):
                redundancyRows.append(
                    {
                        "feature_window": windowName,
                        "fundamental_rule": fundamentalRule,
                        "feature_1": firstName,
                        "feature_2": FEATURE_NAMES[secondIndex],
                        "pearson_r": float(
                            np.corrcoef(
                                values[:, firstIndex],
                                values[:, secondIndex],
                            )[0, 1]
                        ),
                    }
                )

    changeRows = []
    windows = sorted({item[0] for item in configurations})
    for windowName in windows:
        bandKey = (windowName, "band_peak")
        harmonicKey = (windowName, "harmonic_check")
        if bandKey not in featureCache:
            featureCache[bandKey] = calculateFeatures(
                xTrain, windowName=windowName, fundamentalRule="band_peak"
            )
        if harmonicKey not in featureCache:
            featureCache[harmonicKey] = calculateFeatures(
                xTrain,
                windowName=windowName,
                fundamentalRule="harmonic_check",
            )
        bandFrequency = featureCache[bandKey][:, 4]
        harmonicFrequency = featureCache[harmonicKey][:, 4]
        changedIndices = np.flatnonzero(
            ~np.isclose(bandFrequency, harmonicFrequency)
        )
        for sampleIndex in changedIndices:
            changeRows.append(
                {
                    "feature_window": windowName,
                    "sample_index": int(sampleIndex),
                    "voltage_V": float(yTrain[sampleIndex, 0]),
                    "position_cm": float(yTrain[sampleIndex, 1]),
                    "band_peak_Hz": float(bandFrequency[sampleIndex]),
                    "harmonic_check_Hz": float(
                        harmonicFrequency[sampleIndex]
                    ),
                }
            )

    writeRows(
        Path(studyFolder) / "feature_target_correlations.csv",
        correlationRows,
    )
    writeRows(
        Path(studyFolder) / "feature_redundancy.csv",
        redundancyRows,
    )
    writeRows(
        Path(studyFolder) / "fundamental_rule_changes.csv",
        changeRows,
    )


def saveRepeatedShap(studyFolder, xaiRows):
    """Checkpoint per-split and across-split target-specific SHAP evidence."""
    if not xaiRows:
        return
    writeRows(Path(studyFolder) / "feature_shap_repeated.csv", xaiRows)
    grouped = {}
    for row in xaiRows:
        key = (row["experiment_id"], row["feature"], row["target"])
        grouped.setdefault(key, []).append(
            float(row["mean_absolute_shap_value"])
        )
    summaryRows = []
    for (experimentId, featureName, targetName), values in sorted(
        grouped.items()
    ):
        values = np.asarray(values, dtype=np.float64)
        summaryRows.append(
            {
                "experiment_id": experimentId,
                "feature": featureName,
                "target": targetName,
                "repeats": len(values),
                "mean_absolute_shap_value": float(np.mean(values)),
                "split_standard_deviation": (
                    float(np.std(values, ddof=1))
                    if len(values) > 1
                    else 0.0
                ),
            }
        )
    writeRows(
        Path(studyFolder) / "feature_shap_summary.csv",
        summaryRows,
    )
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    featureLabels = {
        "rms_g": "RMS amplitude",
        "peak_to_peak_g": "Peak-to-peak amplitude",
        "crest_factor": "Crest factor",
        "kurtosis": "Kurtosis",
        "dominant_frequency_Hz": "Dominant frequency",
        "dominant_amplitude_g": "Dominant amplitude",
        "high_frequency_power_ratio": "High-frequency ratio",
        "spectral_centroid_Hz": "Spectral centroid",
    }
    targetLabels = {
        "voltage_V": "Voltage",
        "position_cm": "Position",
    }
    for axis, targetName in zip(
        axes, ("voltage_V", "position_cm")
    ):
        rows = [
            row for row in summaryRows if row["target"] == targetName
        ]
        rows.sort(key=lambda row: row["mean_absolute_shap_value"])
        axis.barh(
            [featureLabels[row["feature"]] for row in rows],
            [row["mean_absolute_shap_value"] for row in rows],
            xerr=[row["split_standard_deviation"] for row in rows],
            capsize=3,
        )
        axis.set(
            xlabel="Mean absolute SHAP value",
            ylabel="Feature",
            title=f"SHAP relevance: {targetLabels[targetName]}",
        )
    figure.tight_layout()
    figure.savefig(
        Path(studyFolder) / "feature_shap_summary.png", dpi=200
    )
    plt.close(figure)


def saveRepeatedTimeRelevance(studyFolder, relevanceRows):
    """Save target-specific, across-split grouped time-window evidence."""
    if not relevanceRows:
        return
    writeRows(
        Path(studyFolder) / "time_grouped_relevance_repeated.csv",
        relevanceRows,
    )
    grouped = {}
    for row in relevanceRows:
        key = (
            row["experiment_id"],
            row["target"],
            row["group_number"],
            row["time_centre_s"],
        )
        grouped.setdefault(key, []).append(float(row["relevance"]))
    summaryRows = []
    for (experimentId, targetName, groupNumber, timeCentre), values in sorted(
        grouped.items()
    ):
        values = np.asarray(values, dtype=np.float64)
        summaryRows.append(
            {
                "experiment_id": experimentId,
                "target": targetName,
                "group_number": groupNumber,
                "time_centre_s": timeCentre,
                "repeats": len(values),
                "mean_relevance": float(np.mean(values)),
                "split_standard_deviation": (
                    float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
                ),
            }
        )
    writeRows(
        Path(studyFolder) / "time_grouped_relevance_summary.csv",
        summaryRows,
    )
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharex=True)
    targetLabels = {"voltage_V": "Voltage (V)", "position_cm": "Position (cm)"}
    for axis, targetName in zip(axes, ("voltage_V", "position_cm")):
        rows = [row for row in summaryRows if row["target"] == targetName]
        rows.sort(key=lambda row: row["time_centre_s"])
        axis.plot(
            [row["time_centre_s"] for row in rows],
            [row["mean_relevance"] for row in rows],
            marker="o", markersize=3,
        )
        axis.fill_between(
            [row["time_centre_s"] for row in rows],
            [
                max(0.0, row["mean_relevance"] - row["split_standard_deviation"])
                for row in rows
            ],
            [row["mean_relevance"] + row["split_standard_deviation"] for row in rows],
            alpha=0.2,
        )
        axis.set(
            xlabel="Time-window centre (s)",
            ylabel="Mean absolute prediction change",
            title=(
                "Repeated grouped replacement: "
                f"{targetLabels[targetName]}"
            ),
        )
    figure.tight_layout()
    figure.savefig(
        Path(studyFolder) / "time_grouped_relevance_summary.png", dpi=200
    )
    plt.close(figure)


def saveRepeatedSpectrumRelevance(studyFolder, relevanceRows):
    """Save target-specific, across-split grouped spectral-band evidence."""
    if not relevanceRows:
        return
    writeRows(
        Path(studyFolder) / "spectrum_grouped_relevance_repeated.csv",
        relevanceRows,
    )
    grouped = {}
    for row in relevanceRows:
        key = (
            row["experiment_id"], row["target"], row["group_number"],
            row["frequency_centre_Hz"],
        )
        grouped.setdefault(key, []).append(float(row["relevance"]))
    summaryRows = []
    for (experimentId, targetName, groupNumber, frequencyCentre), values in sorted(grouped.items()):
        values = np.asarray(values, dtype=np.float64)
        summaryRows.append(
            {
                "experiment_id": experimentId,
                "target": targetName,
                "group_number": groupNumber,
                "frequency_centre_Hz": frequencyCentre,
                "repeats": len(values),
                "mean_relevance": float(np.mean(values)),
                "split_standard_deviation": (
                    float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
                ),
            }
        )
    writeRows(
        Path(studyFolder) / "spectrum_grouped_relevance_summary.csv",
        summaryRows,
    )
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharex=True)
    targetLabels = {"voltage_V": "Voltage (V)", "position_cm": "Position (cm)"}
    for axis, targetName in zip(axes, ("voltage_V", "position_cm")):
        rows = [row for row in summaryRows if row["target"] == targetName]
        rows.sort(key=lambda row: row["frequency_centre_Hz"])
        xValues = [row["frequency_centre_Hz"] for row in rows]
        means = [row["mean_relevance"] for row in rows]
        deviations = [row["split_standard_deviation"] for row in rows]
        axis.plot(xValues, means, marker="o", markersize=3)
        axis.fill_between(
            xValues,
            [max(0.0, value - deviation) for value, deviation in zip(means, deviations)],
            [value + deviation for value, deviation in zip(means, deviations)],
            alpha=0.2,
        )
        axis.set(
            xlabel="Frequency-band centre (Hz)",
            ylabel="Mean absolute prediction change",
            title=(
                "Repeated grouped replacement: "
                f"{targetLabels[targetName]}"
            ),
        )
    figure.tight_layout()
    figure.savefig(
        Path(studyFolder) / "spectrum_grouped_relevance_summary.png", dpi=200
    )
    plt.close(figure)


def main():
    args = parseArguments()
    (
        configurationPath,
        configuration,
        experiments,
    ) = loadStudyConfiguration(args.config)

    xTrainRaw, xTestRaw, yTrain = loadData(args.data_dir.resolve())
    _, xTrain = convertADCToAcceleration(xTrainRaw)
    _, xTest = convertADCToAcceleration(xTestRaw)

    splitConfiguration = configuration["splits"]
    seeds = splitConfiguration["seeds"]
    if args.smoke:
        seeds = seeds[:1]
    splits, strata, binsUsed = makeRepeatedTargetAwareSplits(
        yTrain,
        seeds,
        validationFraction=splitConfiguration.get(
            "validation_fraction", 0.2
        ),
        maximumBins=splitConfiguration.get("maximum_target_bins", 4),
    )

    print(
        f"Validated {len(experiments)} experiments across "
        f"{len(splits)} target-aware split(s); "
        f"joint target bins used: {binsUsed}."
    )
    if args.check_only:
        print("Check-only completed; TensorFlow was not imported.")
        return

    import pandas as pd
    import sklearn
    import shap
    import tensorflow as tf

    trainingConfiguration = configuration["training"]
    epochs = 1 if args.smoke else int(trainingConfiguration["epochs"])
    patience = (
        1 if args.smoke else int(trainingConfiguration["patience"])
    )
    batchSize = int(trainingConfiguration["batch_size"])

    studyName = (
        configuration["study_name"] + ("-smoke" if args.smoke else "")
    )
    studyFolder = createImmutableStudyFolder(
        args.output_root, studyName
    )
    print(f"Checkpoint folder: {studyFolder}")
    saveFeatureDiagnostics(studyFolder, xTrain, yTrain, experiments)
    metricRows = []
    predictionRows = []
    xaiRows = []
    timeRelevanceRows = []
    spectrumRelevanceRows = []
    packageVersions = {
        "tensorflow": tf.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "shap": shap.__version__,
    }

    for split in splits:
        seed = split["seed"]
        trainIndices = split["train_indices"]
        validationIndices = split["validation_indices"]
        setReproducibleSeed(seed)
        tf.keras.utils.set_random_seed(seed)
        representations = prepareRepresentations(
            xTrain, xTest, trainIndices
        )
        featureRepresentationCache = {}
        yMean, yStd = fitStandardiser(
            yTrain[trainIndices], axis=0
        )
        yTrainNorm = (yTrain - yMean) / yStd

        for experiment in experiments:
            print(
                f"Repeat {split['repeat'] + 1}/{len(splits)} | "
                f"{experiment.experiment_id}"
            )
            if experiment.method == "feature":
                featureKey = (
                    experiment.feature_window,
                    experiment.fundamental_rule,
                )
                if featureKey not in featureRepresentationCache:
                    rawTrainFeatures = calculateFeatures(
                        xTrain,
                        windowName=experiment.feature_window,
                        fundamentalRule=experiment.fundamental_rule,
                    )
                    rawTestFeatures = calculateFeatures(
                        xTest,
                        windowName=experiment.feature_window,
                        fundamentalRule=experiment.fundamental_rule,
                    )
                    featureMean, featureStd = fitStandardiser(
                        rawTrainFeatures[trainIndices], axis=0
                    )
                    featureRepresentationCache[featureKey] = (
                        (
                            (rawTrainFeatures - featureMean) / featureStd
                        ).astype(np.float32),
                        (
                            (rawTestFeatures - featureMean) / featureStd
                        ).astype(np.float32),
                    )
                xValues, xTestValues = featureRepresentationCache[
                    featureKey
                ]
                featureColumns = selectFeatureColumns(
                    experiment.feature_set
                )
                xValues = xValues[:, featureColumns]
                xTestValues = xTestValues[:, featureColumns]
            else:
                xValues = representations[experiment.method][0]
                xTestValues = representations[experiment.method][1]

            if experiment.model_kind == "neural":
                tf.keras.backend.clear_session()
                tf.keras.utils.set_random_seed(seed)
                model = buildNeuralModel(
                    tf, experiment, xValues.shape[1:], seed
                )
                result = trainAndEvaluateModel(
                    tf=tf,
                    model=model,
                    xTrain=xValues,
                    xTest=xTestValues,
                    yTrainNorm=yTrainNorm,
                    yTrain=yTrain,
                    yMean=yMean,
                    yStd=yStd,
                    trainIndices=trainIndices,
                    validationIndices=validationIndices,
                    epochs=epochs,
                    batchSize=batchSize,
                    patience=patience,
                    predictTest=False,
                    verbose=0,
                    showModelSummary=False,
                )
                metrics = result["metrics"]
                validationPrediction = result["validationPrediction"]
                if experiment.collect_xai and experiment.method == "feature":
                    importance = calculateShapFeatureImportance(
                        result["model"],
                        xValues[trainIndices[:20]],
                        xValues[validationIndices[:20]],
                    )
                    selectedNames = [
                        FEATURE_NAMES[index] for index in featureColumns
                    ]
                    for featureNumber, featureName in enumerate(
                        selectedNames
                    ):
                        for targetNumber, targetName in enumerate(
                            ("voltage_V", "position_cm")
                        ):
                            xaiRows.append(
                                {
                                    "repeat": split["repeat"],
                                    "seed": seed,
                                    "experiment_id": (
                                        experiment.experiment_id
                                    ),
                                    "feature": featureName,
                                    "target": targetName,
                                    "mean_absolute_shap_value": float(
                                        importance["mean"][
                                            featureNumber, targetNumber
                                        ]
                                    ),
                                    "within_split_standard_deviation": float(
                                        importance["std"][
                                            featureNumber, targetNumber
                                        ]
                                    ),
                                    "explained_samples": importance["samples"],
                                }
                            )
                elif experiment.collect_xai and experiment.method == "time":
                    groupCentres, relevance = calculateGroupedReplacementRelevance(
                        result["model"],
                        xValues[validationIndices],
                        numberOfGroups=40,
                        seed=seed,
                    )
                    for targetNumber, targetName in enumerate(
                        ("voltage_V", "position_cm")
                    ):
                        for groupNumber, groupCentre in enumerate(groupCentres):
                            timeRelevanceRows.append(
                                {
                                    "repeat": split["repeat"],
                                    "seed": seed,
                                    "experiment_id": experiment.experiment_id,
                                    "target": targetName,
                                    "group_number": groupNumber,
                                    "time_centre_s": float(groupCentre / 1600.0),
                                    "relevance": float(
                                        relevance[targetNumber, groupNumber]
                                    ),
                                }
                            )
                elif experiment.collect_xai and experiment.method == "spectrum":
                    groupCentres, relevance = calculateGroupedReplacementRelevance(
                        result["model"],
                        xValues[validationIndices],
                        numberOfGroups=50,
                        seed=seed,
                    )
                    frequencyCentres = representations["frequencies"][
                        np.clip(np.rint(groupCentres).astype(int), 0, len(representations["frequencies"]) - 1)
                    ]
                    for targetNumber, targetName in enumerate(
                        ("voltage_V", "position_cm")
                    ):
                        for groupNumber, frequencyCentre in enumerate(frequencyCentres):
                            spectrumRelevanceRows.append(
                                {
                                    "repeat": split["repeat"],
                                    "seed": seed,
                                    "experiment_id": experiment.experiment_id,
                                    "target": targetName,
                                    "group_number": groupNumber,
                                    "frequency_centre_Hz": float(frequencyCentre),
                                    "relevance": float(
                                        relevance[targetNumber, groupNumber]
                                    ),
                                }
                            )
            else:
                metrics, validationPrediction = runClassicalBaseline(
                    experiment,
                    xValues,
                    yTrainNorm,
                    yTrain,
                    yMean,
                    yStd,
                    trainIndices,
                    validationIndices,
                )

            metricRows.append(
                {
                    "repeat": split["repeat"],
                    "seed": seed,
                    "experiment_id": experiment.experiment_id,
                    "method": experiment.method,
                    "model_kind": experiment.model_kind,
                    "architecture": experiment.architecture,
                    "feature_set": experiment.feature_set,
                    "feature_window": experiment.feature_window,
                    "fundamental_rule": experiment.fundamental_rule,
                    "noise_standard_deviation": (
                        experiment.noise_standard_deviation
                    ),
                    **{
                        key: metrics[key]
                        for key in metrics
                        if key != "model"
                    },
                }
            )
            for rowNumber, sampleIndex in enumerate(validationIndices):
                predictionRows.append(
                    {
                        "repeat": split["repeat"],
                        "seed": seed,
                        "experiment_id": experiment.experiment_id,
                        "sample_index": int(sampleIndex),
                        "true_voltage_V": float(yTrain[sampleIndex, 0]),
                        "predicted_voltage_V": float(
                            validationPrediction[rowNumber, 0]
                        ),
                        "true_position_cm": float(yTrain[sampleIndex, 1]),
                        "predicted_position_cm": float(
                            validationPrediction[rowNumber, 1]
                        ),
                    }
                )

            # Checkpoint after every completed experiment. If a long study is
            # interrupted, all completed metrics and predictions remain usable.
            saveRepeatedShap(studyFolder, xaiRows)
            saveRepeatedTimeRelevance(studyFolder, timeRelevanceRows)
            saveRepeatedSpectrumRelevance(studyFolder, spectrumRelevanceRows)
            saveStudyArtifacts(
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
            )
    print(f"Saved report evidence to: {studyFolder}")


if __name__ == "__main__":
    main()
