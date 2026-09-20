"""Section 2 - Feature-based fully connected network.

This file shows only the steps that are specific to the feature approach:

1. calculate physically meaningful time/frequency features,
2. standardise each feature using the shared training subset, and
3. define the fully connected regression network.

Run this method through the common experiment entry point:

    python code/RunAll_Comparison.py --models feature
"""

import numpy as np

from DataAnalysis_Common import (
    FEATURE_NAMES,
    L,
    MOTOR_FUNDAMENTAL_MAX_HZ,
    MOTOR_FUNDAMENTAL_MIN_HZ,
    SPECTRUM_MAX_FREQUENCY_HZ,
    fs,
    fitStandardiser,
)


# =============================================================================
# Section 2A. Extract meaningful signal features
# =============================================================================

def calculateFeatures(
    xAccelerationG,
    windowName="rectangular",
    fundamentalRule="band_peak",
):
    """Calculate eight features using declared, report-testable choices."""
    f = np.fft.rfftfreq(
        L, d=1.0 / fs
    )

    # RUBRIC ADDITION 2026-07-29:
    # Search only the physically possible 20-105 Hz motor-speed range. This
    # prevents the strong 109.6 Hz second harmonic in sample 184 being labelled
    # as the motor fundamental instead of its 54.8 Hz peak.
    fundamentalBand = (
        (f >= MOTOR_FUNDAMENTAL_MIN_HZ)
        & (f <= MOTOR_FUNDAMENTAL_MAX_HZ)
    )
    lowFrequencyPowerBand = (f >= 20.0) & (f <= 110.0)
    highFrequencyBand = (f > 110.0) & (f <= 300.0)
    analysisBand = (
        (f > 0.0) & (f <= SPECTRUM_MAX_FREQUENCY_HZ)
    )

    features = np.empty(
        (xAccelerationG.shape[0], len(FEATURE_NAMES)), dtype=np.float32
    )

    for n, signal in enumerate(xAccelerationG):
        signal64 = signal.astype(np.float64)
        rms = np.sqrt(np.mean(signal64**2))

        if windowName == "rectangular":
            window = np.ones(L, dtype=np.float64)
        elif windowName == "hann":
            window = np.hanning(L)
        else:
            raise ValueError(f"Unknown feature window: {windowName}")

        # Divide by coherent gain so a sinusoid retains its amplitude.
        windowedSignal = signal64 * window
        coherentGain = np.mean(window)
        amplitude = (
            2.0
            * np.abs(np.fft.rfft(windowedSignal))
            / (L * coherentGain)
        )
        fundamentalAmplitude = amplitude[fundamentalBand]
        dominantIndex = int(np.argmax(fundamentalAmplitude))
        dominantFrequency = f[fundamentalBand][dominantIndex]
        dominantAmplitude = fundamentalAmplitude[dominantIndex]

        if fundamentalRule == "harmonic_check":
            halfFrequency = dominantFrequency / 2.0
            if halfFrequency >= MOTOR_FUNDAMENTAL_MIN_HZ:
                halfIndex = int(np.argmin(np.abs(f - halfFrequency)))
                halfAmplitude = amplitude[halfIndex]
                # A visible subharmonic at least one quarter as large as the
                # strongest peak is treated as the rotational fundamental.
                if halfAmplitude >= 0.25 * dominantAmplitude:
                    dominantFrequency = f[halfIndex]
                    dominantAmplitude = halfAmplitude
        elif fundamentalRule != "band_peak":
            raise ValueError(
                f"Unknown fundamental rule: {fundamentalRule}"
            )

        lowFrequencyPower = np.sum(
            amplitude[lowFrequencyPowerBand] ** 2
        )
        spectralPower = amplitude[analysisBand] ** 2

        features[n] = (
            rms,
            np.ptp(signal64),
            np.max(np.abs(signal64)) / (rms + 1e-12),
            np.mean(signal64**4) / (rms**4 + 1e-12),
            dominantFrequency,
            dominantAmplitude,
            np.sum(amplitude[highFrequencyBand] ** 2)
            / (lowFrequencyPower + 1e-12),
            np.sum(f[analysisBand] * spectralPower)
            / (np.sum(spectralPower) + 1e-12),
        )

    return features


# =============================================================================
# Section 2B. Standardise the feature matrix
# =============================================================================

def prepareFeatureInputs(xTrain, xTest, trainIndices):
    """Calculate and standardise features using training data only."""
    xTrainFeatures = calculateFeatures(xTrain)
    xTestFeatures = calculateFeatures(xTest)

    featureMean, featureStd = fitStandardiser(
        xTrainFeatures[trainIndices], axis=0
    )
    xTrainNorm = (
        xTrainFeatures - featureMean
    ) / featureStd
    xTestNorm = (
        xTestFeatures - featureMean
    ) / featureStd

    return (
        xTrainNorm.astype(np.float32),
        xTestNorm.astype(np.float32),
        xTrainFeatures,
    )


# =============================================================================
# Section 2C. Define the feature-based FCN
# =============================================================================

def buildFeatureFCN(
    tf,
    noiseStandardDeviation,
    modelName="feature_fcn",
    layerSeed=None,
    numberOfFeatures=None,
    architecture="adopted",
):
    """Build the feature FCN using the Sequential style from Week 04/05."""
    if numberOfFeatures is None:
        numberOfFeatures = len(FEATURE_NAMES)
    architectureWidths = {
        "small": (16,),
        "adopted": (32, 16),
        "wide": (64, 32),
    }
    if architecture not in architectureWidths:
        raise ValueError(
            f"Unknown feature FCN architecture: {architecture}"
        )
    hiddenLayers = [
        tf.keras.layers.Dense(width, activation="relu")
        for width in architectureWidths[architecture]
    ]
    return tf.keras.Sequential(
        [
            tf.keras.Input((numberOfFeatures,)),
            # This is simply x_augmented = x + random noise during training.
            # STD=0 gives the matched no-augmentation control.
            tf.keras.layers.GaussianNoise(
                noiseStandardDeviation, seed=layerSeed
            ),
            *hiddenLayers,
            tf.keras.layers.Dense(2),
        ],
        name=modelName,
    )
