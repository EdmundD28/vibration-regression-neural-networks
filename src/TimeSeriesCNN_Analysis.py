"""Section 3 - CNN with raw time-series input.

This file shows only the steps specific to the time-series approach:

1. apply one global scale fitted on the shared training subset,
2. preserve the complete 16,000-sample time sequence, and
3. define the one-dimensional convolutional network.

Run this method through the common experiment entry point:

    python code/RunAll_Comparison.py --models time
"""

import numpy as np

from DataAnalysis_Common import L, fitStandardiser


# =============================================================================
# Section 3A. Prepare the raw time-series input
# =============================================================================

def prepareTimeSeriesInputs(xTrain, xTest, trainIndices):
    """Apply one training-fitted global scale and add the channel axis."""
    timeMean, timeStd = fitStandardiser(
        xTrain[trainIndices], axis=(0, 1)
    )

    xTrainNorm = (
        (xTrain - timeMean) / timeStd
    )[..., None]
    xTestNorm = (
        (xTest - timeMean) / timeStd
    )[..., None]

    return (
        xTrainNorm.astype(np.float32),
        xTestNorm.astype(np.float32),
    )


# =============================================================================
# Section 3B. Define the raw time-series CNN
# =============================================================================

def buildTimeSeriesCNN(
    tf,
    noiseStandardDeviation,
    modelName="time_cnn",
    layerSeed=None,
    architecture="adopted",
):
    """Build a declared raw time-series CNN without spectral inputs.

    ``aggressive`` (also the backwards-compatible ``adopted`` label) is the
    original course-style comparator. ``gentle`` retains substantially more
    time positions before global pooling, testing whether extreme downsampling
    discarded useful periodic detail.
    """
    if architecture == "adopted":
        architecture = "aggressive"
    if architecture == "aggressive":
        convolutionPlan = ((16, 33, 4, 4), (32, 17, 2, 4), (64, 9, 2, 4))
    elif architecture == "gentle":
        convolutionPlan = ((16, 65, 2, 2), (32, 33, 1, 2), (64, 17, 1, None))
    else:
        raise ValueError(
            f"Unsupported time-series architecture: {architecture}. "
            "Use 'aggressive' or 'gentle'."
        )

    layers = [
        tf.keras.Input((L, 1)),
        tf.keras.layers.GaussianNoise(
            noiseStandardDeviation, seed=layerSeed
        ),
    ]
    for filters, kernelSize, stride, poolSize in convolutionPlan:
        layers.append(
            tf.keras.layers.Conv1D(
                filters, kernelSize, strides=stride, padding="same",
                activation="relu",
            )
        )
        if poolSize is not None:
            layers.append(tf.keras.layers.MaxPooling1D(poolSize))
    layers.extend(
        [
            tf.keras.layers.GlobalMaxPooling1D(),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(2),
        ]
    )
    return tf.keras.Sequential(
        layers,
        name=modelName,
    )
