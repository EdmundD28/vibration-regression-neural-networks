"""Run the modular vibration analysis and compare the three approaches.

The code order matches the five required report sections:

1. formulate and check the common regression problem,
2. prepare and build the feature FCN,
3. prepare and build the raw time-series CNN,
4. prepare and build the fixed-resolution spectrum network, and
5. train fairly, explain, compare and save the results.

Examples
--------
Check the local data files without importing TensorFlow:

    python src/RunAll_Comparison.py --check-only

Run one method for a short lab demonstration:

    python src/RunAll_Comparison.py --models feature --epochs 2

Run the complete comparison:

    python src/RunAll_Comparison.py

Run a matched Gaussian-noise augmentation study:

    python src/RunAll_Comparison.py --augmentation-study
"""

import argparse
from pathlib import Path
import secrets

import numpy as np

from DataAnalysis_Common import (
    DATA_FOLDER,
    L,
    Ntrain,
    PROJECT_FOLDER,
    calculateGroupedReplacementRelevance,
    calculateShapFeatureImportance,
    convertADCToAcceleration,
    fitStandardiser,
    fs,
    loadData,
    makeTrainValidationSplit,
    saveOutputs,
    setReproducibleSeed,
    trainAndEvaluateModel,
)
from FeatureFCN_Analysis import (
    buildFeatureFCN,
    prepareFeatureInputs,
)
from SpectrumNetwork_Analysis import (
    buildSpectrumNetwork,
    prepareSpectrumInputs,
)
from TimeSeriesCNN_Analysis import (
    buildTimeSeriesCNN,
    prepareTimeSeriesInputs,
)


# =============================================================================
# Command-line settings used for lab demonstrations and full experiments
# =============================================================================

def parseArguments():
    """Read optional settings from the PowerShell command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DATA_FOLDER,
        help="Folder containing the three zID-specific binary files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_FOLDER / "outputs",
        help="Folder for models, metrics, predictions and figures.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=("feature", "time", "spectrum"),
        default=("feature", "time", "spectrum"),
        help="Models to train (default: all three).",
    )
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument(
        "--spectrum-architecture",
        choices=("adopted", "average_pooling", "max_pooling", "flatten",
                 "band_pooling", "compact_flatten", "average_pooling_wide",
                 "flatten_linear", "fine_pooling_linear"),
        default="band_pooling",
        help="Spectrum readout; 'adopted' reproduces the historical global-average baseline.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "Optional run seed. Omit it to generate a new seed and a new "
            "80/20 split for every run; save the printed seed to reproduce "
            "that exact run."
        ),
    )
    parser.add_argument(
        "--noise-augmentation",
        type=float,
        default=0.01,
        metavar="STD",
        help="Training-only Gaussian noise standard deviation; 0 disables it.",
    )
    parser.add_argument(
        "--augmentation-study",
        action="store_true",
        help=(
            "Train matched no-noise control and Gaussian-noise models. "
            "This is optional and does not change the normal run."
        ),
    )
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


# =============================================================================
# Step-by-step analysis in the same order as the report and lab sheets
# =============================================================================

def main():
    """Run the requested methods with one shared experimental protocol."""
    args = parseArguments()
    runSeed = args.seed if args.seed is not None else secrets.randbelow(2**31 - 1)
    setReproducibleSeed(runSeed)

    if args.augmentation_study and args.noise_augmentation <= 0:
        raise ValueError(
            "--augmentation-study requires --noise-augmentation greater than 0."
        )

    # -------------------------------------------------------------------------
    # Section 1: Load, convert and inspect the supplied vibration data
    # -------------------------------------------------------------------------

    xTrainRaw, xTestRaw, yTrain = loadData(args.data_dir.resolve())

    voltageTrain, xTrain = convertADCToAcceleration(xTrainRaw)
    _, xTest = convertADCToAcceleration(xTestRaw)

    trainIndices, validationIndices = makeTrainValidationSplit(
        Ntrain, runSeed
    )

    # Each method prepares its own representation, but all use the same
    # converted signals and the same training/validation sample indices.
    (
        xTrainFeatures,
        xTestFeatures,
        features,
    ) = prepareFeatureInputs(xTrain, xTest, trainIndices)

    xTrainTime, xTestTime = prepareTimeSeriesInputs(
        xTrain, xTest, trainIndices
    )

    (
        xTrainSpectrum,
        xTestSpectrum,
        spectra,
        f,
    ) = prepareSpectrumInputs(xTrain, xTest, trainIndices)

    print("Vibration-regression data check")
    print(f"Run seed: {runSeed}")
    print(
        f"Training/test signals: {xTrain.shape} / {xTest.shape}"
    )
    print(
        f"Targets: {yTrain.shape}; "
        f"voltage {yTrain[:, 0].min():.3f}-"
        f"{yTrain[:, 0].max():.3f} V; "
        f"position {yTrain[:, 1].min():.3f}-"
        f"{yTrain[:, 1].max():.3f} cm"
    )
    print(
        f"Sampling: {fs:.0f} Hz, "
        f"{L / fs:.1f} s, "
        f"{fs / L:.1f} Hz resolution"
    )
    print(
        f"Raw voltage range: {voltageTrain.min():.3f} to "
        f"{voltageTrain.max():.3f} V"
    )
    print(
        f"Feature/spectrum shapes: {features.shape} / "
        f"{spectra.shape} (0-{f[-1]:.0f} Hz)"
    )
    print(
        f"Shared split: {len(trainIndices)} training, "
        f"{len(validationIndices)} validation"
    )

    if args.check_only:
        print(
            "Check-only run completed successfully; "
            "TensorFlow was not imported."
        )
        return

    # TensorFlow is imported only when a network is actually trained.
    try:
        import tensorflow as tf
    except ImportError as error:
        raise RuntimeError(
            "TensorFlow is unavailable. Activate .venv and run: "
            "pip install -r requirements.txt"
        ) from error

    tf.keras.utils.set_random_seed(runSeed)
    try:
        tf.config.experimental.enable_op_determinism()
    except (AttributeError, RuntimeError):
        pass

    # The two engineering targets use different units and numerical scales.
    # Fit one target standardiser on the shared training subset only.
    yTrain = yTrain.astype(np.float32)
    yMean, yStd = fitStandardiser(
        yTrain[trainIndices], axis=0
    )
    yTrainNorm = (yTrain - yMean) / yStd

    # -------------------------------------------------------------------------
    # Sections 2-4: Build the requested method-specific networks
    # -------------------------------------------------------------------------

    modelData = {}

    if "feature" in args.models:
        if args.augmentation_study:
            # RUBRIC ADDITION 2026-07-29:
            # Matched control/augmented models isolate the effect of noise.
            tf.keras.utils.set_random_seed(runSeed)
            featureControl = buildFeatureFCN(
                tf,
                0.0,
                modelName="feature_fcn_control",
                layerSeed=runSeed,
            )
            tf.keras.utils.set_random_seed(runSeed)
            featureAugmented = buildFeatureFCN(
                tf,
                args.noise_augmentation,
                modelName="feature_fcn_augmented",
                layerSeed=runSeed,
            )
            featureAugmented.set_weights(featureControl.get_weights())
            modelData["feature_control"] = (
                featureControl,
                xTrainFeatures,
                xTestFeatures,
                "feature",
                0.0,
            )
            modelData["feature_augmented"] = (
                featureAugmented,
                xTrainFeatures,
                xTestFeatures,
                "feature",
                args.noise_augmentation,
            )
        else:
            myFeatureFCN = buildFeatureFCN(
                tf, args.noise_augmentation
            )
            modelData["feature"] = (
                myFeatureFCN,
                xTrainFeatures,
                xTestFeatures,
                "feature",
                args.noise_augmentation,
            )

    if "time" in args.models:
        if args.augmentation_study:
            tf.keras.utils.set_random_seed(runSeed)
            timeControl = buildTimeSeriesCNN(
                tf,
                0.0,
                modelName="time_cnn_control",
                layerSeed=runSeed,
            )
            tf.keras.utils.set_random_seed(runSeed)
            timeAugmented = buildTimeSeriesCNN(
                tf,
                args.noise_augmentation,
                modelName="time_cnn_augmented",
                layerSeed=runSeed,
            )
            timeAugmented.set_weights(timeControl.get_weights())
            modelData["time_control"] = (
                timeControl,
                xTrainTime,
                xTestTime,
                "time",
                0.0,
            )
            modelData["time_augmented"] = (
                timeAugmented,
                xTrainTime,
                xTestTime,
                "time",
                args.noise_augmentation,
            )
        else:
            myTimeCNN = buildTimeSeriesCNN(
                tf, args.noise_augmentation
            )
            modelData["time"] = (
                myTimeCNN,
                xTrainTime,
                xTestTime,
                "time",
                args.noise_augmentation,
            )

    if "spectrum" in args.models:
        if args.augmentation_study:
            tf.keras.utils.set_random_seed(runSeed)
            spectrumControl = buildSpectrumNetwork(
                tf,
                xTrainSpectrum.shape[1],
                0.0,
                modelName="spectrum_cnn_control",
                layerSeed=runSeed,
                architecture=args.spectrum_architecture,
            )
            tf.keras.utils.set_random_seed(runSeed)
            spectrumAugmented = buildSpectrumNetwork(
                tf,
                xTrainSpectrum.shape[1],
                args.noise_augmentation,
                modelName="spectrum_cnn_augmented",
                layerSeed=runSeed,
                architecture=args.spectrum_architecture,
            )
            spectrumAugmented.set_weights(spectrumControl.get_weights())
            modelData["spectrum_control"] = (
                spectrumControl,
                xTrainSpectrum,
                xTestSpectrum,
                "spectrum",
                0.0,
            )
            modelData["spectrum_augmented"] = (
                spectrumAugmented,
                xTrainSpectrum,
                xTestSpectrum,
                "spectrum",
                args.noise_augmentation,
            )
        else:
            mySpectrumNetwork = buildSpectrumNetwork(
                tf,
                xTrainSpectrum.shape[1],
                args.noise_augmentation,
                architecture=args.spectrum_architecture,
            )
            modelData["spectrum"] = (
                mySpectrumNetwork,
                xTrainSpectrum,
                xTestSpectrum,
                "spectrum",
                args.noise_augmentation,
            )

    # -------------------------------------------------------------------------
    # Section 5A: Train and evaluate every model using the same protocol
    # -------------------------------------------------------------------------

    results = []

    for experimentName in modelData:
        (
            model,
            xTrainModel,
            xTestModel,
            analysisMethod,
            augmentationStandardDeviation,
        ) = modelData[experimentName]

        # RUBRIC ADDITION 2026-07-29:
        # Reset before matched study runs so the comparison is not caused by
        # different initial weights or sample shuffling.
        if args.augmentation_study:
            tf.keras.utils.set_random_seed(runSeed)

        result = trainAndEvaluateModel(
            tf=tf,
            model=model,
            xTrain=xTrainModel,
            xTest=xTestModel,
            yTrainNorm=yTrainNorm,
            yTrain=yTrain,
            yMean=yMean,
            yStd=yStd,
            trainIndices=trainIndices,
            validationIndices=validationIndices,
            epochs=args.epochs,
            batchSize=args.batch_size,
            patience=args.patience,
        )
        result["analysisMethod"] = analysisMethod
        result["augmentationStandardDeviation"] = (
            augmentationStandardDeviation
        )
        result["metrics"]["analysis_method"] = analysisMethod
        result["metrics"]["noise_augmentation_std"] = (
            augmentationStandardDeviation
        )
        results.append(result)

    # -------------------------------------------------------------------------
    # Section 5B: Explain the predictions and save comparison evidence
    # -------------------------------------------------------------------------

    # In an augmentation study the second entry for each method is the
    # augmented model. Use it for XAI while retaining both models in metrics.
    resultByMethod = {
        result["analysisMethod"]: result for result in results
    }

    featureImportance = None
    if "feature" in resultByMethod:
        # Week 07/08 uses a subset for SHAP because the calculation is costly.
        featureImportance = calculateShapFeatureImportance(
            resultByMethod["feature"]["model"],
            xTrainFeatures[trainIndices[:20]],
            xTrainFeatures[validationIndices[:20]],
        )

    groupedRelevance = {}

    if "time" in resultByMethod:
        # Apply the Week 07 replacement idea to 40 time windows. Grouping is
        # necessary because exact SHAP over 16,000 coordinates is impractical.
        timeGroupCentres, timeRelevance = (
            calculateGroupedReplacementRelevance(
                resultByMethod["time"]["model"],
                xTrainTime[validationIndices],
                numberOfGroups=40,
                seed=runSeed,
            )
        )
        groupedRelevance[resultByMethod["time"]["name"]] = (
            timeGroupCentres / fs,
            timeRelevance,
        )

    if "spectrum" in resultByMethod:
        spectrumGroupCentres, spectrumRelevance = (
            calculateGroupedReplacementRelevance(
                resultByMethod["spectrum"]["model"],
                xTrainSpectrum[validationIndices],
                numberOfGroups=50,
                seed=runSeed,
            )
        )
        spectrumCoordinate = np.interp(
            spectrumGroupCentres,
            np.arange(len(f)),
            f,
        )
        groupedRelevance[resultByMethod["spectrum"]["name"]] = (
            spectrumCoordinate,
            spectrumRelevance,
        )

    saveOutputs(
        results,
        args.output_dir,
        trainIndices,
        validationIndices,
        features,
        featureImportance,
        groupedRelevance,
        runSeed,
        args.show,
        spectrumArchitecture=args.spectrum_architecture if "spectrum" in args.models else None,
    )

    # -------------------------------------------------------------------------
    # Section 5C: Print a concise engineering-unit comparison
    # -------------------------------------------------------------------------

    print("\nValidation comparison")
    sortedResults = sorted(
        results,
        key=lambda item: item["metrics"][
            "mean_normalised_rmse"
        ],
    )

    for result in sortedResults:
        metric = result["metrics"]
        print(
            f"{result['name']:24s} | "
            f"V MAE {metric['voltage_V_mae']:.3f} | "
            f"position MAE {metric['position_cm_mae']:.3f} | "
            f"mean NRMSE "
            f"{metric['mean_normalised_rmse']:.3f}"
        )

    print(
        f"Saved reproducible outputs to: "
        f"{args.output_dir.resolve()}"
    )


if __name__ == "__main__":
    main()
