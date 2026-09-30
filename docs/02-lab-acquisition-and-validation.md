# Lab acquisition: ADC, filter, sampling and validation

This public edition is based on Edmund Dai's parameter-selection contribution to the five-author Group 15 *Second Acquisition System Report* (2026 Term 2), with the group's experimental observations clearly attributed as group work. The full group report contains other authors' work and personal identifiers and is not redistributed here.

## From a component proposal to available lab hardware

The lab used an ADXL335 analogue accelerometer, an ADS1015 analogue-to-digital converter and a Raspberry Pi 5. The later motor operating range was 1,200–6,000 rpm, equivalent to a 20–100 Hz fundamental. The design needed to retain this band and useful harmonics while keeping digitised values inside the ADC range.

### ADC input configuration and range

The selected ADS1015 measurement used differential AIN0+ / AIN1− input. This allowed the vibration-related voltage difference to be measured around a smaller baseline than a simple ground-referenced measurement of the accelerometer output. The selected full-scale range was ±2.048 V. For the 12-bit signed conversion used in the analysis, one count represents approximately 1 mV. A narrower range would improve quantisation resolution but could clip the observed waveform; a wider range would waste available codes.

The group report first considered a conservative ±4.096 V single-ended setting, then justified the differential configuration and selected ±2.048 V after lab checks. Its recorded sample spanned approximately −1.58 to +1.28 V in one analysis, remaining inside this ADC range. This observation establishes that **that run** did not clip the ADC. It does not establish that the accelerometer response was perfectly linear; the report separately noted possible sensor-range concerns.

### Analogue filtering

The ADXL335's approximately 32 kΩ internal resistance and a 0.01 µF capacitor give a first-order cutoff near

`f_c = 1/(2πRC) ≈ 497 Hz`.

At 100 Hz the ideal single-pole response retains about 98% of amplitude. A 0.03 µF option would cut off near 166 Hz and attenuate the 100 Hz region more; 0.1 µF would cut off near 50 Hz and suppress part of the required fundamental band. The 0.01 µF choice retained more harmonic information.

This is a **first-order** low-pass filter, not a brick-wall anti-aliasing guarantee. Its cutoff and one observed spectrum do not prove that every possible component above the 800 Hz Nyquist frequency was removed.

### Sampling rate and duration

| Decision | Selected value | Engineering consequence |
|---|---:|---|
| Sampling rate | 1,600 samples/s | Nyquist frequency 800 Hz; margin above the approximately 497 Hz analogue cutoff |
| Duration | 10 s | 16,000 samples per record |
| Frequency-bin spacing | 0.1 Hz | Follows \(\Delta f = 1/T\); one bin corresponds to 6 rpm |
| ADC full-scale range | ±2.048 V | Approximately 1 mV per 12-bit count |
| Input | Differential AIN0+ / AIN1− | Measures the selected voltage difference |

The group downsampled one captured record to 800, 400 and 200 samples/s for comparison. Its 65.3 Hz fundamental and approximately 130 Hz second harmonic remained visible at 800 and 400 samples/s, while the second harmonic exceeded the 100 Hz Nyquist limit of the 200 samples/s version and aliased. This is an empirical illustration of the information lost at too low a rate; it is not a claim that 1,600 samples/s is mathematically optimal for every future rig.

The group also compared 10 s with shorter 5 s and 1 s windows. Bin spacing increased from 0.1 to 0.2 and 1 Hz respectively. The original report described the 5 s spacing as 0.5 Hz; the value here corrects that arithmetic using 1/5 s = 0.2 Hz. A shorter record can still show the main peak, but provides coarser speed discrimination and less evidence about time variation. A 10 s record was retained for the later machine-learning task.

## What the lab measurement changed

One measured case contained a strong 65.3 Hz component and a harmonic around 130 Hz. The earlier one degree of freedom estimate did not reliably predict measured acceleration magnitude. The group described possible accelerometer nonlinearity, mounting and support uncertainty, and residual offset. It recommended a wider linear sensor range in future work. These are warnings against interpreting acceleration amplitude or a model explanation as a proven mechanical sensor position.

The later assigned machine-learning dataset was supplied separately. Its storage format and stated conversion use 330 mV/g, whereas the group's earlier lab discussion used a typical 300 mV/g datasheet value. The later analysis follows its own assignment specification; results from the two stages should not be numerically conflated.

## Hand-off to modelling

The chosen acquisition design gives each model a 10 s, 16,000-sample, 1,600 Hz record. The [model-selection report](03-model-selection-and-stability.md) explains how those same records become engineered features, raw time-series input or a fixed-resolution spectrum.

Source: Group 15, *Second Acquisition System Report*, 2026 Term 2, especially section 2.2 authored by Edmund Dai and the group's section 3 validation. The other contributors were Hiroshi Sogashiwa, Xuehan Sun, Xingjian Bai and Dunxiong Yao. No original group pages, student IDs or third-party figures are reproduced here.
