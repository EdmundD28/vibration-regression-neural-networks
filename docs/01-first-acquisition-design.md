# First acquisition design: the initial engineering estimate

This public edition adapts Edmund Dai's individual *First Acquisition System Report* (2026 Term 2). It preserves the design choices and the limits of the initial calculation without reproducing course handouts or component datasheet pages.

## Question and physical model

The first task was to specify a measurement system for a rotating unbalance mounted at the centre of a 570 × 40 × 3 mm aluminium beam. The motor's original 600–6,000 rpm range corresponds to 10–100 Hz. A one degree of freedom forced vibration approximation was used to estimate the beam response and translate it into sensor range, frequency response, sampling rate and digital resolution requirements.

The model treated the rotating disc's missing mass as an equivalent unbalance and the beam as an equivalent mass, stiffness and damping system. In compact form, its assumed motion is `m q'' + c q' + k q = mₑ e ω² sin(ωt)`. Motor speed sets the excitation frequency; the beam response and measurement location affect amplitude. It predicted a resonance near 26 Hz and a maximum acceleration of about 15.7 m/s², or 1.6 g, at 100 Hz. These values were *design estimates*, not measurements. The assumed support stiffness and 5% damping ratio were uncertain. The later lab measurement exceeded the predicted acceleration substantially, so this model should not be reused as a calibrated amplitude-to-position formula.

## Original component proposal

| Choice | Initial proposal | Reason at the time | What changed later |
|---|---|---|---|
| Sensor | MPU-9250 digital accelerometer | Compact integrated measurement chain | The lab supplied an ADXL335 analogue sensor and ADS1015 ADC |
| Full-scale range | ±2 g | Some margin above the modelled 1.6 g | Lab observations exceeded that estimate; ±2 g was inadequate as a general design target |
| Resolution | 16-bit proposal | Resolve small signals while covering the estimated range | The lab ADC was a 12-bit ADS1015; its input range and clipping had to be checked empirically |
| Signal bandwidth | 184 Hz low-pass proposal | Retain the 10–100 Hz fundamental and suppress higher-frequency noise | The lab design also needed useful harmonic content, so the selected analogue filter had a much higher cutoff |
| Sampling | Approximately 800 Hz proposed | Well above twice the 100 Hz motor fundamental | The lab selected 1,600 samples/s after accounting for the analogue filter bandwidth |

The individual proposal was useful because it turned physical estimates into measurable requirements. It was **not** the hardware used to produce the later data-analysis records. The key lesson is that an early simplified beam model sets a starting range; the measurement system must be revised when real amplitude and harmonic content are observed.

## Connection to the next report

The five-person second acquisition report compared individual proposals, adapted the design to the supplied ADXL335, ADS1015 and Raspberry Pi, and validated the final acquisition settings. Edmund authored its parameter-selection section. The next document records that transition and the lab evidence.

Source: Edmund Dai, *First Acquisition System Report*, 2026 Term 2; Group 15, *Second Acquisition System Report*, 2026 Term 2. The original course files and datasheet appendices are not redistributed here.
