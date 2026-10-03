"""Export a reproducible representative selected spectrum model and its scales.

This is an 80/20 representative model, not a fit on all 200 labelled records.
Accuracy conclusions belong to the repeated study, not this export run.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from DataAnalysis_Common import (DATA_FOLDER, loadData, convertADCToAcceleration,
    setReproducibleSeed, fitStandardiser, trainAndEvaluateModel)
from ExperimentFramework import makeRepeatedTargetAwareSplits, writeRows
from SpectrumNetwork_Analysis import buildSpectrumNetwork, calculateSpectra, prepareSpectrumInputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1053)
    args = parser.parse_args()
    configuration = json.loads(args.config.read_text(encoding="utf-8"))
    candidate = configuration["research_protocol"]["frozen_candidate"]
    spec = next(item for item in configuration["experiments"] if item["id"] == candidate)
    rawTrain, rawTest, targets = loadData(DATA_FOLDER)
    _, signals = convertADCToAcceleration(rawTrain)
    _, testSignals = convertADCToAcceleration(rawTest)
    split = makeRepeatedTargetAwareSplits(targets, [args.seed], validationFraction=0.2, maximumBins=4)[0][0]
    trainingIndices, validationIndices = split["train_indices"], split["validation_indices"]
    setReproducibleSeed(args.seed)
    import tensorflow as tf
    tf.keras.utils.set_random_seed(args.seed)
    trainInputs, testInputs, spectra, frequencies = prepareSpectrumInputs(signals, testSignals, trainingIndices)
    spectrumMean, spectrumStd = fitStandardiser(spectra[trainingIndices], axis=(0, 1))
    targetMean, targetStd = fitStandardiser(targets[trainingIndices], axis=0)
    model = buildSpectrumNetwork(tf, trainInputs.shape[1], spec["noise_standard_deviation"],
        modelName=candidate, layerSeed=args.seed, architecture=spec["architecture"])
    settings = configuration["training"]
    result = trainAndEvaluateModel(tf, model, trainInputs, testInputs,
        (targets - targetMean) / targetStd, targets, targetMean, targetStd,
        trainingIndices, validationIndices, settings["epochs"], settings["batch_size"], settings["patience"],
        predictTest=False, verbose=0, showModelSummary=False)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    modelFile = args.output_dir / "spectrum_position_model.keras"
    model.save(modelFile)
    np.savez(args.output_dir / "spectrum_position_scales.npz", spectrum_mean=spectrumMean,
             spectrum_std=spectrumStd, target_mean=targetMean, target_std=targetStd,
             frequencies_hz=frequencies, training_indices=trainingIndices, validation_indices=validationIndices)
    restored = tf.keras.models.load_model(modelFile)
    np.testing.assert_allclose(model(trainInputs[validationIndices], training=False).numpy(),
        restored(trainInputs[validationIndices], training=False).numpy(), atol=1e-6)
    # Also verify the saved scales reproduce the complete inference pipeline.
    saved = np.load(args.output_dir / "spectrum_position_scales.npz")
    validationSpectra, _ = calculateSpectra(signals[validationIndices])
    restoredInputs = ((validationSpectra - saved["spectrum_mean"]) / saved["spectrum_std"])[..., None]
    restoredPrediction = restored(restoredInputs, training=False).numpy() * saved["target_std"] + saved["target_mean"]
    np.testing.assert_allclose(restoredPrediction, result["validationPrediction"], atol=1e-5)
    exportManifest = {"experiment": spec, "seed": args.seed, "training_records": len(trainingIndices),
        "validation_records": len(validationIndices), "configuration_file": str(args.config.resolve()),
        "metrics": result["metrics"], "history": result["history"],
        "serialization_and_scales_verified": True,
        "model_scope": "Representative 80/20 model; not trained on all labelled records.",
        "inference": "ADC counts -> per-record mean-centred acceleration g -> rFFT amplitude 2*abs(rFFT)/16000 -> retain 0..500 Hz -> log1p -> saved global spectrum scale -> model -> saved target inverse scale"}
    (args.output_dir / "export_manifest.json").write_text(json.dumps(exportManifest, indent=2), encoding="utf-8")
    writeRows(args.output_dir / "representative_validation_predictions.csv", [
        {"sample_index": int(index), "true_voltage_V": float(targets[index, 0]),
         "predicted_voltage_V": float(result["validationPrediction"][number, 0]),
         "true_position_cm": float(targets[index, 1]),
         "predicted_position_cm": float(result["validationPrediction"][number, 1])}
        for number, index in enumerate(validationIndices)])
    print(json.dumps({"model": str(modelFile.resolve()), "metrics": result["metrics"], "verified": True}, indent=2))


if __name__ == "__main__":
    main()
