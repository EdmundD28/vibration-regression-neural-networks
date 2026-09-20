"""Section 4 - Network with fixed-resolution spectrum input.

This file shows only the steps specific to the spectral approach:

1. calculate a proper 0-500 Hz spectrum at 0.1 Hz resolution,
2. apply log-amplitude preprocessing and one training-fitted scale, and
3. define the one-dimensional spectral convolutional network.

Only spectral amplitude is supplied to the model; frequency is retained only
for interpretation and plotting.

Run this method through the common experiment entry point:

    python code/RunAll_Comparison.py --models spectrum
"""

import numpy as np

from DataAnalysis_Common import (
    L,
    SPECTRUM_MAX_FREQUENCY_HZ,
    fs,
    fitStandardiser,
)


# =============================================================================
# Section 4A. Calculate proper fixed-resolution spectra
# =============================================================================

def calculateSpectra(xAccelerationG):
    """Calculate 0-500 Hz log-amplitude spectra at 0.1 Hz resolution."""
    f = np.fft.rfftfreq(
        L, d=1.0 / fs
    )
    frequencyMask = f <= SPECTRUM_MAX_FREQUENCY_HZ

    amplitude = (
        2.0
        * np.abs(np.fft.rfft(xAccelerationG, axis=1))
        / L
    )
    spectra = np.log1p(amplitude[:, frequencyMask]).astype(np.float32)

    return spectra, f[frequencyMask].astype(np.float32)


# =============================================================================
# Section 4B. Standardise the spectral input
# =============================================================================

def prepareSpectrumInputs(xTrain, xTest, trainIndices):
    """Calculate spectra and apply one training-fitted global scale."""
    xTrainSpectra, f = calculateSpectra(xTrain)
    xTestSpectra, _ = calculateSpectra(xTest)

    spectrumMean, spectrumStd = fitStandardiser(
        xTrainSpectra[trainIndices], axis=(0, 1)
    )
    xTrainNorm = (
        (xTrainSpectra - spectrumMean) / spectrumStd
    )[..., None]
    xTestNorm = (
        (xTestSpectra - spectrumMean) / spectrumStd
    )[..., None]

    return (
        xTrainNorm.astype(np.float32),
        xTestNorm.astype(np.float32),
        xTrainSpectra,
        f,
    )


# =============================================================================
# Section 4C. Define the spectrum CNN
# =============================================================================

def buildSpectrumNetwork(
    tf,
    numberOfFrequencyBins,
    noiseStandardDeviation,
    modelName="spectrum_cnn",
    layerSeed=None,
    architecture="adopted",
):
    """Build a declared fixed-resolution spectrum CNN.

    ``average_pooling`` (and the backwards-compatible ``adopted`` label)
    retains distributed harmonic evidence.  ``max_pooling`` is the matched
    alternative used to test whether one dominant spectral region is enough.
    """
    if architecture == "adopted":
        architecture = "average_pooling"
    if architecture == "average_pooling":
        globalPooling = tf.keras.layers.GlobalAveragePooling1D()
    elif architecture == "max_pooling":
        globalPooling = tf.keras.layers.GlobalMaxPooling1D()
    else:
        raise ValueError(
            f"Unsupported spectrum architecture: {architecture}. "
            "Use 'average_pooling' or 'max_pooling'."
        )

    return tf.keras.Sequential(
        [
            tf.keras.Input((numberOfFrequencyBins, 1)),
            tf.keras.layers.GaussianNoise(
                noiseStandardDeviation, seed=layerSeed
            ),
            tf.keras.layers.Conv1D(
                16, 21, padding="same", activation="relu"
            ),
            tf.keras.layers.MaxPooling1D(4),
            tf.keras.layers.Conv1D(
                32, 11, padding="same", activation="relu"
            ),
            tf.keras.layers.MaxPooling1D(4),
            tf.keras.layers.Conv1D(
                64, 7, padding="same", activation="relu"
            ),
            tf.keras.layers.MaxPooling1D(4),
            globalPooling,
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(2),
        ],
        name=modelName,
    )
