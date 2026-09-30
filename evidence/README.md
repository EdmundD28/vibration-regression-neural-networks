# Evidence index

The published numbers come from five fixed target-aware 80/20 validation splits with seeds 53, 153, 253, 353 and 453. Each split had 160 training and 40 validation records. All input and target scaling was fitted only on the training records. The 50 hidden-label test records do not appear in these metrics.

| Folder | Decision and files | Main conclusion |
|---|---|---|
| [feature-fcn](feature-fcn/) | Corrected feature and architecture selection; `summary_metrics.csv`, `run_metrics.csv`, selection figure | Adopt the seven-feature, 32–16, 818-parameter FCN. The selected experiment ID is `rect_drop_crest_factor`. |
| [time-cnn](time-cnn/) | Aggressive/gentle architecture control, noise control, final summary, residual and relevance figures | The aggressive raw-time CNN achieved the best position RMSE; zero training noise was selected. |
| [spectrum-cnn](spectrum-cnn/) | Average/max pooling control, noise control, final summary and relevance figure | Average pooling with training noise 0.01 was selected, but clean performance remained weaker. |
| [gain-stress](gain-stress/) | Shared clean and +5% held-out gain comparison | A small simulated calibration shift changed the time CNN less than the competitive feature FCN. |

The CSVs retain original experiment IDs, split-level metrics and summary statistics. They do not contain original signal records, hidden-test labels or trained scalers. The original study manifests included local absolute paths and are not published; [experiment JSON files](../experiments/) state the relevant configurations.

The original report's repeated SHAP figure was generated in a **wide 2,658-parameter feature-model comparator** run. It should not be represented as an explanation of the selected 818-parameter FCN. Accordingly, it is omitted from the final selected-model gallery.
