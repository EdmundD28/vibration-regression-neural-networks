# Trained models

These Keras models come from the single showcased run recorded in `../results/run_manifest.json`.

- `feature_fcn.keras`: fully connected network using eight engineered vibration features.
- `time_cnn.keras`: one-dimensional convolutional network using raw time-series input.
- `spectrum_cnn.keras`: one-dimensional convolutional network using a fixed-resolution amplitude spectrum.

The files contain trained networks, not a complete inference product. Correct use requires the matching signal conversion, representation construction and training-fitted standardisation implemented in `../src/`.

Do not treat these models as calibrated diagnostic or safety systems.
