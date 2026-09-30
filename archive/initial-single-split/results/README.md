# Training and validation evidence

This folder is a compact, auditable snapshot of one training run.

`model_comparison.csv` is the authoritative metric table. The run used seed `1612835101`, 160 training samples and 40 validation samples. Model selection from a single split can be optimistic, so these results should not be presented as uncertainty-bounded population performance.

The figures retain the most useful evidence for portfolio review:

- `validation_loss.png`
- `training_validation_predictions.png`
- `residual_diagnostics.png`
- `feature_shap_importance.png`
- `time_cnn_grouped_relevance.png`
- `spectrum_cnn_grouped_relevance.png`
- `split_target_coverage.png`

Raw source data and hidden-test predictions are intentionally not published.
