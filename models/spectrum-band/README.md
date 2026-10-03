# Representative band-aggregation spectrum model

This is the selected 47,138-parameter architecture exported from a **160-training / 40-validation** fit with seed 1053. It is not a fit on all 200 records and is not an externally validated deployment model. Repeated-study means are reported in [the readout evidence](../../evidence/spectrum-readout/), rather than inferred from this representative fit.

`spectrum_position_model.keras` and `spectrum_position_scales.npz` form the complete inference artifact. The scale archive stores the spectrum mean/standard deviation, target means/standard deviations, frequency grid and the training/validation indices. The export manifest records the fit and confirms save/reload agreement for both the model and the full preprocessing pipeline.

Inference: ADC counts → per-record mean-centred acceleration using 0.330 V/g → `2*abs(rFFT)/16000` amplitude → retain 0–500 Hz → `log1p` → saved global spectrum scale → model → saved target inverse scale. The two outputs are voltage in V and position in cm.

To reproduce the representative fit with authorised local data:

```powershell
python src/ExportSpectrumPositionModel.py --config experiments/spectrum_position_confirmation_study.json --output-dir outputs/spectrum-band-export --seed 1053
```

Historical gain-robustness and frequency-band explanation results describe the old global-average model. They have not been repeated for these weights.
