# Evidence index

The original published numbers come from five fixed target-aware 80/20 validation splits with seeds 53, 153, 253, 353 and 453. Each split had 160 training and 40 validation records. All input and target scaling was fitted only on the training records. The 50 hidden-label test records do not appear in these metrics.

| Folder | Decision and files | Main conclusion |
|---|---|---|
| [feature-fcn](feature-fcn/) | Corrected feature and architecture selection; `summary_metrics.csv`, `run_metrics.csv`, selection figure | Adopt the seven-feature, 32–16, 818-parameter FCN. The selected experiment ID is `rect_drop_crest_factor`. |
| [time-cnn](time-cnn/) | Aggressive/gentle architecture control, noise control, final summary, residual and relevance figures | The aggressive raw-time CNN achieved the best position RMSE; zero training noise was selected. |
| [spectrum-cnn](spectrum-cnn/) | Historical average/max pooling control, noise control, summary and relevance figure | These results describe the old global-average model, superseded by the revised readout. |
| [spectrum-readout](spectrum-readout/) | 60 model/split runs: readout selection, capacity/local-pooling candidates, frozen confirmation; metrics, split assignments, figures and adapted provenance | Adopt band aggregation (47,138 parameters): confirmation NRMSE 0.252, 57.4% below the old model; position comparable to raw time, joint error still 9.4% higher. |
| [gain-stress](gain-stress/) | Shared clean and +5% held-out gain comparison | A small simulated calibration shift changed the time CNN less than the competitive feature FCN. |

The CSVs retain original experiment IDs, split-level metrics and summary statistics. They do not contain original signal records or hidden-test labels. The separate representative-model folder contains its fitted scalers. The historical study manifests included local absolute paths and are not published. The revised readout manifests and provenance omit those paths and preserve original-run hashes with an adaptation note; [experiment JSON files](../experiments/) state the relevant configurations.

The original report's repeated SHAP figure was generated in a **wide 2,658-parameter feature-model comparator** run. It should not be represented as an explanation of the selected 818-parameter FCN. Accordingly, it is omitted from the final selected-model gallery.

The revised confirmation stage uses seeds 1053, 1153, 1253, 1353 and 1453 after freezing band aggregation on the original selection seeds. These split assignments reuse the same 200 labelled recordings, and validation also controls early stopping. Confirmation checks stability, not independent external-test accuracy or significance. Historical gain stress and spectrum relevance remain specific to the old model.
