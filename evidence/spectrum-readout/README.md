# Frequency-position readout investigation

The amplitude-spectrum input is unchanged: full 10 s rFFT, 0–500 Hz at 0.1 Hz resolution, `log1p` amplitude and training-fitted global scaling. The standard spectrum backbone has Conv1D 16/32/64, kernels 21/11/7 and three local MaxPool(4) stages, producing 78 × 64 responses.

- `selection/`: six models × original five seeds (30 runs), comparing global average, full flatten, ordered band aggregation, compact flatten, wide global-average capacity control and raw time.
- `capacity/`: two extra candidates × the same five seeds (10 runs), testing a direct linear head and finer local pooling.
- `confirmation/`: frozen band aggregation, both global-average controls and raw time × five new seeds (20 runs). No retuning after confirmation.

Band aggregation uses AveragePool(6)-Flatten, retaining 13 × 64 ordered responses before Dense(32)-Dense(2). It has 47,138 parameters; noise 0.01 is held fixed across spectrum candidates. `research_summary.json` and `comparison.csv` keep selection and confirmation statistics separate. The full Chinese research account is in `research_report.html`.

Confirmation mean NRMSE was 0.25175 versus 0.59049 for the old model (57.4% lower) and 0.23020 for raw time (9.4% higher). Position RMSE was 1.43383 versus 1.48690 cm for raw time; voltage RMSE remained 0.24173 versus 0.18142 V. Position had three of five paired wins, voltage zero of five, and joint error one of five. The declared 5% overall-parity criterion was not met.

The unchanged input and weak 180,280-parameter global-average control support frequency-location aggregation as an important bottleneck, without establishing a sole cause or a need for phase. Repeated overlapping holdouts, also used for early stopping, are stability checks rather than external-test or statistical-significance evidence. Old gain/XAI results are specific to the old model.

Public manifests omit private local paths and original labelled per-record prediction files. Original source/data hashes are retained as provenance and explicitly refer to the original local run, not adapted public source. Private source snapshots are preserved in the local archive. The published configurations and current source reproduce the declared experiment choices with authorised local data.
