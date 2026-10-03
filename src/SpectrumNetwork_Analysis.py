"""Section 4 - Network with fixed-resolution spectrum input.

This file shows only the steps specific to the spectral approach:

1. calculate a proper 0-500 Hz spectrum at 0.1 Hz resolution,
2. apply log-amplitude preprocessing and one training-fitted scale, and
3. define a one-dimensional network retaining frequency-band positions.

Only spectral amplitude is supplied to the model; frequency is retained only
for interpretation and plotting.

Run this method through the common experiment entry point:

    python src/RunAll_Comparison.py --models spectrum
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
    architecture="band_pooling",
):
    """Build a declared fixed-resolution spectrum CNN.

    ``adopted`` retains the historical average-pooling baseline so archived
    study configurations remain reproducible. Position-preserving readouts
    keep the same input, noise layer and three convolution/pooling blocks:
    ``flatten`` keeps every remaining frequency position; ``band_pooling``
    averages six neighbouring positions before flattening; ``compact_flatten``
    projects channels to four before flattening. ``average_pooling_wide`` is
    a parameter-count control for ``flatten`` at the project's 5,001 bins.
    ``flatten_linear`` deletes the hidden dense layer; ``fine_pooling_linear``
    also reduces each local pooling width to two. The current default is the
    confirmed ``band_pooling`` readout (47,138 parameters).
    """
    if architecture == "adopted":
        architecture = "average_pooling"
    localPoolSize = 2 if architecture == "fine_pooling_linear" else 4
    backboneLayers = [
        tf.keras.Input((numberOfFrequencyBins, 1)),
        tf.keras.layers.GaussianNoise(noiseStandardDeviation, seed=layerSeed),
        tf.keras.layers.Conv1D(16, 21, padding="same", activation="relu"),
        tf.keras.layers.MaxPooling1D(localPoolSize),
        tf.keras.layers.Conv1D(32, 11, padding="same", activation="relu"),
        tf.keras.layers.MaxPooling1D(localPoolSize),
        tf.keras.layers.Conv1D(64, 7, padding="same", activation="relu"),
        tf.keras.layers.MaxPooling1D(localPoolSize),
    ]
    denseWidth = 32
    if architecture == "average_pooling":
        readoutLayers = [tf.keras.layers.GlobalAveragePooling1D()]
    elif architecture == "max_pooling":
        readoutLayers = [tf.keras.layers.GlobalMaxPooling1D()]
    elif architecture in ("flatten", "flatten_linear", "fine_pooling_linear"):
        readoutLayers = [tf.keras.layers.Flatten()]
    elif architecture == "band_pooling":
        readoutLayers = [
            tf.keras.layers.AveragePooling1D(6),
            tf.keras.layers.Flatten(),
        ]
    elif architecture == "compact_flatten":
        readoutLayers = [
            tf.keras.layers.Conv1D(4, 1, activation="relu"),
            tf.keras.layers.Flatten(),
        ]
    elif architecture == "average_pooling_wide":
        readoutLayers = [tf.keras.layers.GlobalAveragePooling1D()]
        # Match the flatten head's parameter budget to the closest integer.
        remainingPositions = int(numberOfFrequencyBins) // (4 ** 3)
        denseWidth = max(1, round((remainingPositions * 64 * 32 + 96) / 67))
    else:
        raise ValueError(
            f"Unsupported spectrum architecture: {architecture}. "
            "Use 'average_pooling', 'max_pooling', 'flatten', 'band_pooling', "
            "'compact_flatten', 'average_pooling_wide', 'flatten_linear' "
            "or 'fine_pooling_linear'."
        )

    predictionLayers = (
        [tf.keras.layers.Dense(2)]
        if architecture in ("flatten_linear", "fine_pooling_linear")
        else [tf.keras.layers.Dense(denseWidth, activation="relu"), tf.keras.layers.Dense(2)]
    )

    return tf.keras.Sequential(
        [
            *backboneLayers,
            *readoutLayers,
            *predictionLayers,
        ],
        name=modelName,
    )
