"""Verify frequency-position retention, matched backbones and Keras round trips."""
import json
from pathlib import Path
import tempfile

import numpy as np
import tensorflow as tf

from SpectrumNetwork_Analysis import buildSpectrumNetwork


def main():
    expected = {
        "average_pooling": (22562, 64),
        "max_pooling": (22562, 64),
        "flatten": (180258, 4992),
        "band_pooling": (47138, 832),
        "compact_flatten": (30758, 312),
        "average_pooling_wide": (180280, 64),
        "flatten_linear": (30402, 4992),
        "fine_pooling_linear": (100418, 40000),
    }
    referenceWeights = None
    rows = []
    inputs = np.random.default_rng(53).normal(size=(2, 5001, 1)).astype("float32")
    for architecture, (parameters, width) in expected.items():
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(53)
        model = buildSpectrumNetwork(tf, 5001, 0.01, layerSeed=53, architecture=architecture)
        assert model.count_params() == parameters, architecture
        assert model.output_shape == (None, 2), architecture
        linearHead = architecture in {"flatten_linear", "fine_pooling_linear"}
        readoutEnd = -1 if linearHead else -2
        assert model.layers[readoutEnd - 1].output.shape[-1] == width, architecture
        backboneWeights = [weight.numpy() for layer in model.layers[:7] for weight in layer.weights]
        if referenceWeights is None:
            referenceWeights = backboneWeights
        else:
            for reference, actual in zip(referenceWeights, backboneWeights, strict=True):
                np.testing.assert_array_equal(reference, actual)
        prediction = model(inputs, training=False).numpy()
        np.testing.assert_array_equal(prediction, model(inputs, training=False).numpy())
        with tempfile.TemporaryDirectory() as folder:
            modelFile = Path(folder) / "model.keras"
            model.save(modelFile)
            restored = tf.keras.models.load_model(modelFile)
            np.testing.assert_allclose(prediction, restored(inputs, training=False).numpy(), atol=1e-6)
        # Apply the real post-backbone readout to synthetic feature maps.
        numberOfPositions = int(model.layers[6].output.shape[1])
        featureInput = tf.keras.Input((numberOfPositions, 64))
        features = featureInput
        for layer in model.layers[7:readoutEnd]:
            features = layer(features)
        readout = tf.keras.Model(featureInput, features)
        first = np.zeros((1, numberOfPositions, 64), dtype="float32")
        first[:, 12, :] = 10
        shifted = np.zeros_like(first)
        shifted[:, 60, :] = 10
        firstResult = readout(first).numpy()
        shiftedResult = readout(shifted).numpy()
        positionRetained = not np.allclose(firstResult, shiftedResult)
        assert positionRetained == (architecture in {"flatten", "band_pooling", "compact_flatten", "flatten_linear", "fine_pooling_linear"}), architecture
        rows.append({"architecture": architecture, "parameters": parameters, "readout_width": width,
                     "frequency_shift_distinguished": positionRetained, "serialization_verified": True})
    print(json.dumps({"status": "passed", "architectures": rows}, indent=2))


if __name__ == "__main__":
    main()
