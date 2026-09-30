"""Run one shared sensor-gain stress test for the selected Section 2-4 models.

All models are trained on the unchanged training fold.  The held-out record is
then multiplied by 1.05 before its method-specific preprocessing and
prediction.  This is an out-of-distribution calibration-drift stress test, not
a claim about the measured sensor-noise distribution.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from DataAnalysis_Common import (
    DATA_FOLDER,
    PROJECT_FOLDER,
    calculateRegressionMetrics,
    convertADCToAcceleration,
    fitStandardiser,
    loadData,
    setReproducibleSeed,
    trainAndEvaluateModel,
)
from ExperimentFramework import (
    createImmutableStudyFolder,
    makeRepeatedTargetAwareSplits,
    writeRows,
)
from FeatureFCN_Analysis import buildFeatureFCN, calculateFeatures
from SpectrumNetwork_Analysis import buildSpectrumNetwork, calculateSpectra
from TimeSeriesCNN_Analysis import buildTimeSeriesCNN


SEEDS = (53, 153, 253, 353, 453)
GAIN = 1.05
EPOCHS = 200
BATCH_SIZE = 16
PATIENCE = 20


def feature_inputs(x, train_indices, validation_indices):
    clean = calculateFeatures(x, windowName="rectangular", fundamentalRule="band_peak")
    mean, std = fitStandardiser(clean[train_indices], axis=0)
    clean = ((clean - mean) / std).astype(np.float32)
    stressed_raw = calculateFeatures(
        x[validation_indices] * GAIN,
        windowName="rectangular",
        fundamentalRule="band_peak",
    )
    stressed = ((stressed_raw - mean) / std).astype(np.float32)
    # The selected Section 2 model excludes crest factor (column 2).
    columns = tuple(index for index in range(clean.shape[1]) if index != 2)
    return clean[:, columns], stressed[:, columns]


def time_inputs(x, train_indices, validation_indices):
    mean, std = fitStandardiser(x[train_indices], axis=(0, 1))
    clean = ((x - mean) / std)[..., None].astype(np.float32)
    stressed = (((x[validation_indices] * GAIN) - mean) / std)[..., None]
    return clean, stressed.astype(np.float32)


def spectrum_inputs(x, train_indices, validation_indices):
    clean_raw, _ = calculateSpectra(x)
    mean, std = fitStandardiser(clean_raw[train_indices], axis=(0, 1))
    clean = ((clean_raw - mean) / std)[..., None].astype(np.float32)
    stressed_raw, _ = calculateSpectra(x[validation_indices] * GAIN)
    stressed = ((stressed_raw - mean) / std)[..., None]
    return clean, stressed.astype(np.float32)


def main():
    import matplotlib.pyplot as plt
    import pandas as pd
    import tensorflow as tf

    x_raw, _, y = loadData(DATA_FOLDER)
    _, x = convertADCToAcceleration(x_raw)
    splits, _, _ = makeRepeatedTargetAwareSplits(y, SEEDS, validationFraction=0.2, maximumBins=4)
    study_folder = createImmutableStudyFolder(
        PROJECT_FOLDER / "outputs" / "studies", "section5-shared-gain-stress"
    )
    models = (
        ("feature_fcn", feature_inputs, lambda shape, seed: buildFeatureFCN(tf, 0.0, "section5_feature", seed, shape[0], "adopted")),
        ("time_cnn", time_inputs, lambda shape, seed: buildTimeSeriesCNN(tf, 0.0, "section5_time", seed, "aggressive")),
        ("spectrum_cnn", spectrum_inputs, lambda shape, seed: buildSpectrumNetwork(tf, shape[0], 0.01, "section5_spectrum", seed, "average_pooling")),
    )
    rows = []
    for split in splits:
        seed = split["seed"]
        train_indices = split["train_indices"]
        validation_indices = split["validation_indices"]
        y_mean, y_std = fitStandardiser(y[train_indices], axis=0)
        y_norm = (y - y_mean) / y_std
        for model_id, prepare, build in models:
            print(f"Repeat {split['repeat'] + 1}/5 | {model_id}", flush=True)
            setReproducibleSeed(seed)
            tf.keras.backend.clear_session()
            tf.keras.utils.set_random_seed(seed)
            clean_inputs, stressed_validation = prepare(x, train_indices, validation_indices)
            model = build(clean_inputs.shape[1:], seed)
            result = trainAndEvaluateModel(
                tf, model, clean_inputs, clean_inputs, y_norm, y, y_mean, y_std,
                train_indices, validation_indices, EPOCHS, BATCH_SIZE, PATIENCE,
                predictTest=False, verbose=0, showModelSummary=False,
            )
            clean_metrics = result["metrics"]
            stressed_prediction = model.predict(stressed_validation, verbose=0) * y_std + y_mean
            stressed_metrics = calculateRegressionMetrics(y[validation_indices], stressed_prediction, y_std)
            for condition, metrics in (("clean", clean_metrics), ("gain_plus_5_percent", stressed_metrics)):
                rows.append({
                    "repeat": split["repeat"], "seed": seed, "condition": condition,
                    **metrics, "model": model_id,
                })
    frame = pd.DataFrame(rows)
    frame.to_csv(study_folder / "run_metrics.csv", index=False)
    summary = frame.groupby(["model", "condition"], as_index=False).agg(
        mean_normalised_rmse_mean=("mean_normalised_rmse", "mean"),
        mean_normalised_rmse_sd=("mean_normalised_rmse", "std"),
        voltage_rmse_V_mean=("voltage_V_rmse", "mean"),
        position_rmse_cm_mean=("position_cm_rmse", "mean"),
    )
    clean = summary[summary.condition == "clean"].set_index("model")
    stressed = summary[summary.condition == "gain_plus_5_percent"].set_index("model")
    nrmse_change_by_model = (
        stressed["mean_normalised_rmse_mean"]
        - clean["mean_normalised_rmse_mean"]
    )
    summary["mean_normalised_rmse_change"] = summary["model"].map(
        nrmse_change_by_model
    )
    summary.to_csv(study_folder / "summary_metrics.csv", index=False)
    pivot = summary.pivot(index="model", columns="condition", values="mean_normalised_rmse_mean")
    pivot = pivot.rename(
        index={
            "feature_fcn": "Feature FCN",
            "time_cnn": "Raw-time CNN",
            "spectrum_cnn": "Spectrum CNN",
        },
        columns={
            "clean": "Clean validation",
            "gain_plus_5_percent": "+5% gain validation",
        },
    )
    pivot = pivot.reindex(
        ["Feature FCN", "Raw-time CNN", "Spectrum CNN"]
    )
    axis = pivot.plot.bar(
        color=["#4C78A8", "#F58518"],
        rot=0,
    )
    axis.set(
        xlabel="Model",
        ylabel="Mean normalised RMSE",
        title="Calibration-gain stress test",
    )
    axis.legend(title="Validation condition")
    plt.tight_layout()
    plt.savefig(study_folder / "gain_stress_summary.png", dpi=180)
    print(summary.to_string(index=False))
    print(f"Saved Section 5 gain-stress evidence to: {study_folder}")


if __name__ == "__main__":
    main()
