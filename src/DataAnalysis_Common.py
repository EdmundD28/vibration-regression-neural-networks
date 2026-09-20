"""Shared vibration-regression data, training and reporting functions.

This file contains only the steps that must remain identical for the three
neural-network approaches:

1. load and validate the binary datasets,
2. convert ADC counts to acceleration,
3. create the shared training/validation split,
4. standardise using training data only,
5. train and evaluate with the same protocol, and
6. save directly comparable report evidence.

The method-specific input preparation and network definitions are kept in:

    FeatureFCN_Analysis.py
    TimeSeriesCNN_Analysis.py
    SpectrumNetwork_Analysis.py
"""

import csv
import json
import os
from pathlib import Path
import random
import time

import numpy as np


# =============================================================================
# 1. Data and acquisition settings
# =============================================================================

CODE_FOLDER = Path(__file__).resolve().parent
PROJECT_FOLDER = CODE_FOLDER.parent
DATA_FOLDER = PROJECT_FOLDER / "data"
TRAINING_SIGNALS_FILENAME = "training_signals.bin"
TESTING_SIGNALS_FILENAME = "testing_signals.bin"
TRAINING_TARGETS_FILENAME = "training_targets.bin"

# Compact symbols used throughout the analysis:
# Ntrain/Ntest = number of signals, L = samples per signal, fs = sample rate.
Ntrain = 200
Ntest = 50
L = 16_000
fs = 1_600.0

ADC_RANGE_V = 2.048
ADC_POSITIVE_COUNTS = 2_048.0
ACCELEROMETER_SENSITIVITY_V_PER_G = 0.330

SPECTRUM_MAX_FREQUENCY_HZ = 500.0
RANDOM_SEED = 53

# Physics-informed frequency bounds:
# The rig operates at 1,200-6,000 rpm (20-100 Hz). A small margin allows
# speed variation without admitting the clear 109.6 Hz second harmonic.
MOTOR_FUNDAMENTAL_MIN_HZ = 20.0
MOTOR_FUNDAMENTAL_MAX_HZ = 105.0

TARGET_NAMES = ("voltage_V", "position_cm")
FEATURE_NAMES = (
    "rms_g",
    "peak_to_peak_g",
    "crest_factor",
    "kurtosis",
    "dominant_frequency_Hz",
    "dominant_amplitude_g",
    "high_frequency_power_ratio",
    "spectral_centroid_Hz",
)


# =============================================================================
# 2. Reproducibility and data loading
# =============================================================================

def setReproducibleSeed(seed=RANDOM_SEED):
    """Use the same pseudo-random sequence every time the analysis is run."""
    os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_FOLDER / ".matplotlib"))
    random.seed(seed)
    np.random.seed(seed)


def loadData(dataFolder):
    """Load the three binary datasets and check their sizes and values."""
    trainingSignalFile = dataFolder / TRAINING_SIGNALS_FILENAME
    testingSignalFile = dataFolder / TESTING_SIGNALS_FILENAME
    trainingTargetFile = dataFolder / TRAINING_TARGETS_FILENAME

    requiredFiles = {
        "training signals": trainingSignalFile,
        "testing signals": testingSignalFile,
        "training targets": trainingTargetFile,
    }
    missingFiles = [
        f"{label}: {path}"
        for label, path in requiredFiles.items()
        if not path.is_file()
    ]
    if missingFiles:
        raise FileNotFoundError(
            "Missing required data file(s):\n  " + "\n  ".join(missingFiles)
        )

    xTrainFlat = np.fromfile(trainingSignalFile, dtype=np.int16)
    xTestFlat = np.fromfile(testingSignalFile, dtype=np.int16)
    yTrainFlat = np.fromfile(trainingTargetFile, dtype=np.float64)

    expectedSizes = (
        Ntrain * L,
        Ntest * L,
        Ntrain * 2,
    )
    actualSizes = (xTrainFlat.size, xTestFlat.size, yTrainFlat.size)
    if actualSizes != expectedSizes:
        raise ValueError(
            f"Unexpected binary file sizes: expected {expectedSizes}, "
            f"got {actualSizes}"
        )

    # As in the Week 05 solution, each row is one complete signal.
    xTrainRaw = xTrainFlat.reshape(Ntrain, L)
    xTestRaw = xTestFlat.reshape(Ntest, L)
    yTrain = yTrainFlat.reshape(Ntrain, 2)

    if not np.all(np.isfinite(yTrain)):
        raise ValueError("Training targets contain NaN or infinity.")

    for name, values in (("training", xTrainRaw), ("testing", xTestRaw)):
        if values.min() < -2048 or values.max() > 2047:
            raise ValueError(
                f"{name.capitalize()} ADC values exceed the signed 12-bit range."
            )

    return xTrainRaw, xTestRaw, yTrain


def convertADCToAcceleration(xRaw):
    """Convert signed ADC counts to volts and mean-centred acceleration in g."""
    voltage = xRaw.astype(np.float32) * (
        ADC_RANGE_V / ADC_POSITIVE_COUNTS
    )
    accelerationG = (
        voltage - voltage.mean(axis=1, keepdims=True)
    ) / ACCELEROMETER_SENSITIVITY_V_PER_G
    return voltage, accelerationG.astype(np.float32)


# =============================================================================
# 3. Shared validation and scaling protocol
# =============================================================================

def makeTrainValidationSplit(Nsamples, seed):
    """Create one shared, seed-controlled 80/20 split for a run."""
    indices = np.random.default_rng(seed).permutation(Nsamples)
    splitIndex = int(0.8 * Nsamples)
    return indices[:splitIndex], indices[splitIndex:]


def fitStandardiser(trainingData, axis=0):
    """Find mean and standard deviation from training data only."""
    mean = np.mean(trainingData, axis=axis, keepdims=True)
    std = np.std(trainingData, axis=axis, keepdims=True)
    std = np.maximum(std, 1e-8)
    return mean.astype(np.float32), std.astype(np.float32)


def calculateRegressionMetrics(yTrue, yPredicted, yTrainingStd):
    """Calculate report metrics in the original engineering units."""
    error = yPredicted - yTrue
    metrics = {}

    for column, targetName in enumerate(TARGET_NAMES):
        residualSum = np.sum(error[:, column] ** 2)
        totalSum = np.sum(
            (yTrue[:, column] - np.mean(yTrue[:, column])) ** 2
        )

        metrics[f"{targetName}_mae"] = float(
            np.mean(np.abs(error[:, column]))
        )
        metrics[f"{targetName}_rmse"] = float(
            np.sqrt(np.mean(error[:, column] ** 2))
        )
        metrics[f"{targetName}_r2"] = float(
            1.0 - residualSum / (totalSum + 1e-12)
        )

    errorNorm = error / yTrainingStd.reshape(1, 2)
    metrics["mean_normalised_rmse"] = float(
        np.mean(np.sqrt(np.mean(errorNorm**2, axis=0)))
    )

    return metrics


# =============================================================================
# 4. Common model training and prediction
# =============================================================================

def trainAndEvaluateModel(
    tf,
    model,
    xTrain,
    xTest,
    yTrainNorm,
    yTrain,
    yMean,
    yStd,
    trainIndices,
    validationIndices,
    epochs,
    batchSize,
    patience,
    predictTest=True,
    verbose=2,
    showModelSummary=True,
):
    """Train one model using the common protocol and return its results."""
    print(
        f"\nTraining {model.name} "
        f"({model.count_params():,} parameters)"
    )
    # The Week 07/08 solutions print the layers before compiling the model.
    if showModelSummary:
        model.summary()

    # Regression uses mean squared error instead of classification loss.
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="mse",
        metrics=["mae"],
    )

    callbacks = [
        # Stop when validation no longer improves and keep the best weights.
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=max(3, patience // 3),
            min_lr=1e-5,
        ),
    ]

    startTime = time.perf_counter()
    history = model.fit(
        xTrain[trainIndices],
        yTrainNorm[trainIndices],
        validation_data=(
            xTrain[validationIndices],
            yTrainNorm[validationIndices],
        ),
        epochs=epochs,
        batch_size=batchSize,
        callbacks=callbacks,
        verbose=verbose,
        shuffle=True,
    )
    trainTime = time.perf_counter() - startTime

    # "Hat" follows the lab-sheet convention for an estimated output.
    yTrainHatNorm = model.predict(
        xTrain[trainIndices], verbose=0
    )
    yValidationHatNorm = model.predict(
        xTrain[validationIndices], verbose=0
    )
    yTestHatNorm = (
        model.predict(xTest, verbose=0)
        if predictTest
        else None
    )

    yTrainHat = yTrainHatNorm * yStd + yMean
    yValidationHat = yValidationHatNorm * yStd + yMean
    yTestHat = (
        yTestHatNorm * yStd + yMean
        if yTestHatNorm is not None
        else None
    )

    metrics = calculateRegressionMetrics(
        yTrain[validationIndices],
        yValidationHat,
        yStd,
    )

    # Physics-informed frequency bounds:
    # Save training metrics too. Their difference from validation metrics is
    # direct quantitative evidence of underfitting or overfitting.
    trainingMetrics = calculateRegressionMetrics(
        yTrain[trainIndices],
        yTrainHat,
        yStd,
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
            "model": model.name,
            "parameters": int(model.count_params()),
            "epochs_run": len(history.history["loss"]),
            "training_seconds": float(trainTime),
        }
    )

    return {
        "name": model.name,
        "model": model,
        "history": history.history,
        "metrics": metrics,
        "trainingTrue": yTrain[trainIndices],
        "trainingPrediction": yTrainHat,
        "validationTrue": yTrain[validationIndices],
        "validationPrediction": yValidationHat,
        "testPrediction": yTestHat,
    }


# =============================================================================
# 5. Explainable AI calculations
# =============================================================================

def calculateShapFeatureImportance(
    model,
    backgroundSamples,
    explainedSamples,
):
    """Apply the SHAP workflow demonstrated in Week 07/08."""
    try:
        import shap
    except ImportError as error:
        raise RuntimeError(
            "SHAP is required for the course XAI method. Run: "
            "pip install -r requirements.txt"
        ) from error

    # Week 07 wraps model.predict in a normal Python function before passing
    # it to shap.Explainer. Small subsets keep the calculation tractable.
    def shapFunction(x):
        return model.predict(x, verbose=0)

    explainer = shap.Explainer(shapFunction, backgroundSamples)
    explanation = explainer(explainedSamples)
    absoluteShapValues = np.abs(np.asarray(explanation.values))

    if absoluteShapValues.ndim == 2:
        absoluteShapValues = absoluteShapValues[..., None]
    if absoluteShapValues.shape[1:] != (
        backgroundSamples.shape[1],
        len(TARGET_NAMES),
    ):
        raise ValueError(
            "Unexpected SHAP value shape: "
            f"{absoluteShapValues.shape}"
        )

    return {
        "mean": np.mean(absoluteShapValues, axis=0),
        "std": np.std(absoluteShapValues, axis=0),
        "samples": len(explainedSamples),
    }


def calculateGroupedReplacementRelevance(
    model,
    samples,
    numberOfGroups,
    numberOfRepeats=5,
    seed=RANDOM_SEED,
):
    """Use Week 07-style feature replacement on grouped signal coordinates."""
    baselinePrediction = model.predict(samples, verbose=0)
    groupEdges = np.linspace(
        0,
        samples.shape[1],
        numberOfGroups + 1,
        dtype=int,
    )
    relevanceRepeats = np.empty(
        (
            numberOfRepeats,
            numberOfGroups,
            len(TARGET_NAMES),
        ),
        dtype=np.float64,
    )
    randomGenerator = np.random.default_rng(seed)

    # Week 07 explains that a feature can be replaced with the corresponding
    # feature from a randomly selected sample. Adjacent coordinates are grouped
    # so 16,000 time samples do not require 16,000 explanations.
    for repeatNumber in range(numberOfRepeats):
        donorIndices = randomGenerator.permutation(len(samples))
        for groupNumber in range(numberOfGroups):
            start = groupEdges[groupNumber]
            stop = groupEdges[groupNumber + 1]
            changedSamples = samples.copy()
            changedSamples[:, start:stop] = samples[
                donorIndices, start:stop
            ]
            changedPrediction = model.predict(
                changedSamples, verbose=0
            )
            relevanceRepeats[
                repeatNumber, groupNumber
            ] = np.mean(
                np.abs(changedPrediction - baselinePrediction),
                axis=0,
            )

    groupCentres = (
        groupEdges[:-1] + groupEdges[1:] - 1
    ) / 2.0
    return (
        groupCentres,
        np.mean(relevanceRepeats, axis=0).T,
    )


# =============================================================================
# 6. Tables, models and figures used as report evidence
# =============================================================================

def saveOutputs(
    results,
    outputFolder,
    trainIndices,
    validationIndices,
    featureValues,
    featureImportance,
    groupedRelevance,
    runSeed,
    showFigures,
):
    """Save models, metrics, predictions and comparison figures."""
    outputFolder.mkdir(parents=True, exist_ok=True)

    for result in results:
        result["model"].save(
            outputFolder / f"{result['name']}.keras"
        )
        np.savetxt(
            outputFolder / f"{result['name']}_test_predictions.csv",
            result["testPrediction"],
            delimiter=",",
            header=",".join(TARGET_NAMES),
            comments="",
        )

        historyFile = outputFolder / f"{result['name']}_history.json"
        historyValues = {
            key: [float(value) for value in values]
            for key, values in result["history"].items()
        }
        with historyFile.open("w", encoding="utf-8") as file:
            json.dump(historyValues, file, indent=2)

    metricKeys = list(results[0]["metrics"].keys())
    with (outputFolder / "model_comparison.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=metricKeys)
        writer.writeheader()
        writer.writerows(result["metrics"] for result in results)

    # Physics-informed frequency bounds:
    # If matched control/augmented results are present, save their direct
    # differences instead of asking the report writer to subtract tables.
    augmentationRows = []
    studyMetrics = (
        "voltage_V_mae",
        "voltage_V_rmse",
        "voltage_V_r2",
        "position_cm_mae",
        "position_cm_rmse",
        "position_cm_r2",
        "mean_normalised_rmse",
        "generalisation_gap_mean_normalised_rmse",
    )
    methods = sorted(
        {
            result.get("analysisMethod")
            for result in results
            if result.get("analysisMethod") is not None
        }
    )
    for method in methods:
        methodResults = [
            result
            for result in results
            if result.get("analysisMethod") == method
        ]
        controlResults = [
            result
            for result in methodResults
            if result.get("augmentationStandardDeviation") == 0
        ]
        augmentedResults = [
            result
            for result in methodResults
            if result.get("augmentationStandardDeviation", 0) > 0
        ]
        if not controlResults or not augmentedResults:
            continue

        control = controlResults[0]
        augmented = augmentedResults[0]
        for metricName in studyMetrics:
            controlValue = control["metrics"][metricName]
            augmentedValue = augmented["metrics"][metricName]
            augmentationRows.append(
                {
                    "analysis_method": method,
                    "metric": metricName,
                    "control": controlValue,
                    "augmented": augmentedValue,
                    "augmented_minus_control": (
                        augmentedValue - controlValue
                    ),
                    "noise_augmentation_std": augmented[
                        "augmentationStandardDeviation"
                    ],
                }
            )

    if augmentationRows:
        with (outputFolder / "augmentation_comparison.csv").open(
            "w", newline="", encoding="utf-8"
        ) as file:
            writer = csv.DictWriter(
                file, fieldnames=augmentationRows[0].keys()
            )
            writer.writeheader()
            writer.writerows(augmentationRows)

    np.savetxt(
        outputFolder / "training_indices.csv",
        trainIndices,
        fmt="%d",
    )
    np.savetxt(
        outputFolder / "validation_indices.csv",
        validationIndices,
        fmt="%d",
    )
    with (outputFolder / "run_manifest.json").open("w", encoding="utf-8") as file:
        json.dump(
            {
                "run_seed": int(runSeed),
                "training_samples": len(trainIndices),
                "validation_samples": len(validationIndices),
                "split": "random 80/20 permutation using run_seed",
            },
            file,
            indent=2,
        )

    # Physics-informed frequency bounds:
    # Save every calculated feature with its split and target values so feature
    # claims in the report can be checked against the exact source samples.
    splitLabel = np.full(len(featureValues), "", dtype=object)
    splitLabel[trainIndices] = "training"
    splitLabel[validationIndices] = "validation"
    targetsByOriginalIndex = np.empty(
        (len(featureValues), len(TARGET_NAMES)), dtype=np.float32
    )
    referenceResult = results[0]
    targetsByOriginalIndex[trainIndices] = referenceResult["trainingTrue"]
    targetsByOriginalIndex[validationIndices] = (
        referenceResult["validationTrue"]
    )
    with (outputFolder / "feature_values.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.writer(file)
        writer.writerow(
            ("sample_index", "split", *TARGET_NAMES, *FEATURE_NAMES)
        )
        for sampleIndex in range(len(featureValues)):
            writer.writerow(
                (
                    sampleIndex,
                    splitLabel[sampleIndex],
                    *targetsByOriginalIndex[sampleIndex],
                    *featureValues[sampleIndex],
                )
            )

    if featureImportance is not None:
        with (outputFolder / "feature_shap_importance.csv").open(
            "w", newline="", encoding="utf-8"
        ) as file:
            writer = csv.writer(file)
            writer.writerow(
                (
                    "feature",
                    "target",
                    "mean_absolute_shap_value",
                    "standard_deviation",
                    "explained_samples",
                )
            )
            # Physics-informed frequency bounds:
            # Keep target-specific XAI evidence in a report-ready table.
            for featureNumber, featureName in enumerate(FEATURE_NAMES):
                for targetNumber, targetName in enumerate(TARGET_NAMES):
                    writer.writerow(
                        (
                            featureName,
                            targetName,
                            featureImportance["mean"][
                                featureNumber, targetNumber
                            ],
                            featureImportance["std"][
                                featureNumber, targetNumber
                            ],
                            featureImportance["samples"],
                        )
                    )

    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(
        len(results),
        2,
        figsize=(9, 3.6 * len(results)),
        squeeze=False,
    )
    plotLimits = ((2.0, 6.0), (5.0, 23.0))

    for row, result in enumerate(results):
        for column, targetName in enumerate(TARGET_NAMES):
            axes[row, column].scatter(
                result["trainingTrue"][:, column],
                result["trainingPrediction"][:, column],
                s=22,
                color="#1f77b4",
                alpha=0.65,
                label=(
                    f"Training (80%, "
                    f"n={len(result['trainingTrue'])})"
                ),
            )
            axes[row, column].scatter(
                result["validationTrue"][:, column],
                result["validationPrediction"][:, column],
                s=28,
                color="#ff7f0e",
                alpha=0.85,
                label=(
                    f"Validation (20%, "
                    f"n={len(result['validationTrue'])})"
                ),
            )
            axes[row, column].plot(
                plotLimits[column],
                plotLimits[column],
                "k--",
                linewidth=1,
            )
            axes[row, column].set(
                xlabel=f"True {targetName}",
                ylabel=f"Predicted {targetName}",
                title=result["name"],
            )
            axes[row, column].legend()

    figure.tight_layout()
    figure.savefig(
        outputFolder / "training_validation_predictions.png", dpi=200
    )
    # This plot contains both training and validation points.  The old
    # validation_predictions.png name was a byte-for-byte duplicate, so remove
    # the stale legacy output rather than producing two indistinguishable files.
    (outputFolder / "validation_predictions.png").unlink(missing_ok=True)

    figure, axis = plt.subplots(figsize=(7, 4))
    for result in results:
        axis.plot(
            result["history"]["val_loss"], label=result["name"]
        )
    axis.set(
        xlabel="Epoch",
        ylabel="Validation MSE (normalised targets)",
        yscale="log",
    )
    axis.legend()
    figure.tight_layout()
    figure.savefig(outputFolder / "validation_loss.png", dpi=200)

    # Physics-informed frequency bounds:
    # Show whether the fixed 80/20 split covers similar target ranges.
    figure, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    referenceResult = results[0]
    for column, targetName in enumerate(TARGET_NAMES):
        allTargetValues = np.concatenate(
            (
                referenceResult["trainingTrue"][:, column],
                referenceResult["validationTrue"][:, column],
            )
        )
        binEdges = np.linspace(
            np.min(allTargetValues), np.max(allTargetValues), 13
        )
        # Paired histograms must use side-by-side bars, not overlapping bars:
        # within every bin, training is blue on the left and validation is
        # orange on the right.  Reuse this pattern for future two-series
        # histogram comparisons so neither series hides the other.
        trainingDensity, _ = np.histogram(
            referenceResult["trainingTrue"][:, column],
            bins=binEdges,
            density=True,
        )
        validationDensity, _ = np.histogram(
            referenceResult["validationTrue"][:, column],
            bins=binEdges,
            density=True,
        )
        # Use ordinal bin positions instead of a continuous x-axis.  This
        # makes each blue/orange pair visibly belong together, with a distinct
        # gap before the next pair.  The label below every pair states its
        # target-value interval.
        binPositions = np.arange(len(binEdges) - 1)
        barWidth = 0.42
        binLabels = [
            f"{start:.2f}-{stop:.2f}"
            for start, stop in zip(binEdges[:-1], binEdges[1:])
        ]
        axes[column].bar(
            binPositions - barWidth / 2,
            trainingDensity,
            width=barWidth,
            color="#1f77b4",
            label="Training (80%)",
        )
        axes[column].bar(
            binPositions + barWidth / 2,
            validationDensity,
            width=barWidth,
            color="#ff7f0e",
            label="Validation (20%)",
        )
        axes[column].set(
            xlabel=(
                "Voltage interval (V)"
                if targetName == "voltage_V"
                else "Position interval (cm)"
            ),
            ylabel="Probability density",
            title=f"{targetName} coverage",
        )
        axes[column].set_xticks(binPositions, binLabels, rotation=45, ha="right")
        axes[column].legend()
    figure.tight_layout()
    figure.savefig(outputFolder / "split_target_coverage.png", dpi=200)

    # Physics-informed frequency bounds:
    # Residual plots expose bias and target-range failure that one MAE cannot.
    figure, axes = plt.subplots(
        len(results),
        2,
        figsize=(9, 3.6 * len(results)),
        squeeze=False,
    )
    for row, result in enumerate(results):
        for column, targetName in enumerate(TARGET_NAMES):
            trainingResidual = (
                result["trainingPrediction"][:, column]
                - result["trainingTrue"][:, column]
            )
            validationResidual = (
                result["validationPrediction"][:, column]
                - result["validationTrue"][:, column]
            )
            axes[row, column].scatter(
                result["trainingTrue"][:, column],
                trainingResidual,
                s=22,
                color="#1f77b4",
                alpha=0.60,
                label="Training (80%)",
            )
            axes[row, column].scatter(
                result["validationTrue"][:, column],
                validationResidual,
                s=28,
                color="#ff7f0e",
                alpha=0.85,
                label="Validation (20%)",
            )
            axes[row, column].axhline(
                0.0, color="black", linestyle="--", linewidth=1
            )
            axes[row, column].set(
                xlabel=f"True {targetName}",
                ylabel="Prediction residual",
                title=f"{result['name']} - {targetName}",
            )
            axes[row, column].legend()
    figure.tight_layout()
    figure.savefig(outputFolder / "residual_diagnostics.png", dpi=200)

    if featureImportance is not None:
        # Physics-informed frequency bounds:
        # Two panels prevent voltage and position relevance being conflated.
        figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        for targetNumber, targetName in enumerate(TARGET_NAMES):
            importanceMean = featureImportance["mean"][:, targetNumber]
            importanceStd = featureImportance["std"][:, targetNumber]
            order = np.argsort(importanceMean)
            axes[targetNumber].barh(
                np.asarray(FEATURE_NAMES)[order],
                importanceMean[order],
                xerr=importanceStd[order],
            )
            axes[targetNumber].set(
                xlabel="Mean absolute SHAP value",
                title=f"SHAP relevance: {targetName}",
            )
        figure.tight_layout()
        figure.savefig(
            outputFolder / "feature_shap_importance.png",
            dpi=200,
        )

    for modelName, (coordinate, values) in groupedRelevance.items():
        np.savetxt(
            outputFolder / f"{modelName}_grouped_relevance.csv",
            np.column_stack((coordinate, values.T)),
            delimiter=",",
            header="coordinate," + ",".join(TARGET_NAMES),
            comments="",
        )
        figure, axis = plt.subplots(figsize=(8, 3.5))
        for targetNumber, targetName in enumerate(TARGET_NAMES):
            axis.plot(
                coordinate,
                values[targetNumber],
                linewidth=0.8,
                label=targetName,
            )
        axis.set(
            xlabel=(
                "Time (s)"
                if modelName.startswith("time_cnn")
                else "Frequency (Hz)"
            ),
            ylabel="Mean absolute prediction change",
            title=f"{modelName} grouped feature relevance",
        )
        axis.legend()
        figure.tight_layout()
        figure.savefig(
            outputFolder / f"{modelName}_grouped_relevance.png",
            dpi=200,
        )

    if showFigures:
        plt.show()
    else:
        plt.close("all")
