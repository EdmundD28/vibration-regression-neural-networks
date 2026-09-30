"""Audit the declared band-peak frequency feature without pre-written results.

This script does not decide a mechanical fundamental from FFT amplitude alone.
It tests whether the declared 20--105 Hz band-peak *feature rule* is consistent
with the known motor-voltage labels, and whether an earlier half-frequency
heuristic creates worse frequency--voltage consistency for the records it
changes. Every reported sample number, frequency, count and conclusion is
computed from the loaded binary data.
"""

import csv
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from DataAnalysis_Common import (
    DATA_FOLDER,
    L,
    MOTOR_FUNDAMENTAL_MAX_HZ,
    MOTOR_FUNDAMENTAL_MIN_HZ,
    PROJECT_FOLDER,
    fs,
    convertADCToAcceleration,
    loadData,
)


OUTPUT_FOLDER = PROJECT_FOLDER / "outputs" / "studies" / "fundamental-frequency-audit"
HARMONIC_TOLERANCE_HZ = fs / L  # One FFT bin.


def spectrum_amplitude(signal):
    """Return the one-sided amplitude spectrum of one mean-centred record."""
    frequencies = np.fft.rfftfreq(L, d=1.0 / fs)
    amplitudes = 2.0 * np.abs(np.fft.rfft(signal.astype(np.float64))) / L
    return frequencies, amplitudes


def peak_index(frequencies, amplitudes, lower_hz, upper_hz):
    """Return the FFT index of the largest amplitude in an inclusive band."""
    in_band = (frequencies >= lower_hz) & (frequencies <= upper_hz)
    indices = np.flatnonzero(in_band)
    if not len(indices):
        raise ValueError(f"Empty frequency band {lower_hz}--{upper_hz} Hz")
    return int(indices[np.argmax(amplitudes[in_band])])


def leave_one_out_prediction(voltage, band_peak_hz, index):
    """Predict one frequency from voltage using the other 199 band-peak values."""
    keep = np.arange(len(voltage)) != index
    slope, intercept = np.polyfit(voltage[keep], band_peak_hz[keep], 1)
    return float(slope * voltage[index] + intercept)


def format_hz(value):
    return f"{value:.1f} Hz"


def main():
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    raw_train, _, targets = loadData(DATA_FOLDER)
    _, acceleration = convertADCToAcceleration(raw_train)

    rows = []
    for sample_index, signal in enumerate(acceleration):
        frequencies, amplitudes = spectrum_amplitude(signal)
        band_index = peak_index(
            frequencies,
            amplitudes,
            MOTOR_FUNDAMENTAL_MIN_HZ,
            MOTOR_FUNDAMENTAL_MAX_HZ,
        )
        broad_index = peak_index(frequencies, amplitudes, MOTOR_FUNDAMENTAL_MIN_HZ, 250.0)
        band_frequency = float(frequencies[band_index])
        band_amplitude = float(amplitudes[band_index])
        broad_frequency = float(frequencies[broad_index])
        broad_amplitude = float(amplitudes[broad_index])
        harmonic_order = int(np.rint(broad_frequency / band_frequency))
        harmonic_error = abs(broad_frequency - harmonic_order * band_frequency)
        broad_is_integer_harmonic = harmonic_order >= 2 and harmonic_error <= HARMONIC_TOLERANCE_HZ

        half_index = int(np.argmin(np.abs(frequencies - band_frequency / 2.0)))
        half_frequency = float(frequencies[half_index])
        half_amplitude = float(amplitudes[half_index])
        historical_rule_changed = (
            half_frequency >= MOTOR_FUNDAMENTAL_MIN_HZ
            and half_amplitude >= 0.25 * band_amplitude
        )
        rows.append(
            {
                "sample_index": sample_index,
                "voltage_V": float(targets[sample_index, 0]),
                "position_cm": float(targets[sample_index, 1]),
                "band_peak_Hz": band_frequency,
                "band_peak_amplitude_g": band_amplitude,
                "broad_peak_Hz": broad_frequency,
                "broad_peak_amplitude_g": broad_amplitude,
                "broad_to_band_amplitude_ratio": broad_amplitude / (band_amplitude + 1e-12),
                "broad_peak_harmonic_order": harmonic_order,
                "broad_peak_harmonic_error_Hz": harmonic_error,
                "broad_peak_is_integer_harmonic": broad_is_integer_harmonic,
                "historical_half_rule_Hz": half_frequency if historical_rule_changed else band_frequency,
                "historical_rule_changed": historical_rule_changed,
            }
        )

    voltage = np.asarray([row["voltage_V"] for row in rows])
    band_peak = np.asarray([row["band_peak_Hz"] for row in rows])
    correlation = float(np.corrcoef(voltage, band_peak)[0, 1])
    slope, intercept = np.polyfit(voltage, band_peak, 1)
    changed_rows = [row for row in rows if row["historical_rule_changed"]]
    for row in changed_rows:
        index = row["sample_index"]
        expected = leave_one_out_prediction(voltage, band_peak, index)
        row["leave_one_out_expected_Hz"] = expected
        row["band_peak_absolute_residual_Hz"] = abs(row["band_peak_Hz"] - expected)
        row["half_rule_absolute_residual_Hz"] = abs(row["historical_half_rule_Hz"] - expected)
        row["half_rule_is_worse"] = (
            row["half_rule_absolute_residual_Hz"] > row["band_peak_absolute_residual_Hz"]
        )
    false_changes = [row for row in changed_rows if row["half_rule_is_worse"]]

    second_harmonic_cases = [
        row
        for row in rows
        if row["broad_peak_Hz"] > MOTOR_FUNDAMENTAL_MAX_HZ
        and row["broad_peak_is_integer_harmonic"]
        and row["broad_peak_harmonic_order"] == 2
    ]
    if not second_harmonic_cases:
        raise RuntimeError("No out-of-band second-harmonic case was found; no audit conclusion is emitted.")
    harmonic_case = max(
        second_harmonic_cases,
        key=lambda row: row["broad_to_band_amplitude_ratio"],
    )

    audit_passed = (
        correlation >= 0.95
        and len(changed_rows) > 0
        and len(false_changes) == len(changed_rows)
        and harmonic_case["broad_peak_harmonic_order"] == 2
    )
    if audit_passed:
        conclusion = (
            "PASS: Retain the declared 20--105 Hz band-peak feature rule. "
            f"It has Pearson r={correlation:.3f} with the known voltage labels. "
            f"The automatically selected out-of-band second-harmonic case is sample {harmonic_case['sample_index']}: "
            f"the selected {format_hz(harmonic_case['band_peak_Hz'])} candidate and "
            f"the broad-band {format_hz(harmonic_case['broad_peak_Hz'])} peak have "
            f"an automatically detected order-{harmonic_case['broad_peak_harmonic_order']} relation. "
            f"The historical half-frequency heuristic changes {len(changed_rows)} record(s), and "
            f"all {len(false_changes)} changed record(s) have a larger leave-one-out frequency--voltage "
            "residual after halving."
        )
    else:
        conclusion = (
            "FAIL: The declared band-peak rule was not confirmed by the predeclared audit criteria. "
            f"correlation={correlation:.3f}, historical_changes={len(changed_rows)}, "
            f"worse_after_halving={len(false_changes)}, "
            f"harmonic_order={harmonic_case['broad_peak_harmonic_order']}."
        )

    summary = {
        "rule_under_test": "Largest FFT amplitude within the declared 20--105 Hz motor-speed band",
        "interpretation_boundary": (
            "This is an audit of a feature-extraction convention against labelled data; "
            "it does not independently prove a mechanical fundamental."
        ),
        "predeclared_pass_criteria": {
            "minimum_band_peak_voltage_pearson_r": 0.95,
            "all_historical_half_rule_changes_worse_under_leave_one_out_fit": True,
        "at_least_one_detected_out_of_band_case_is_second_harmonic": True,
        },
        "samples": len(rows),
        "band_peak_voltage_pearson_r": correlation,
        "linear_frequency_voltage_fit_Hz": {
            "slope_per_V": float(slope),
            "intercept": float(intercept),
        },
        "automatically_selected_second_harmonic_case": harmonic_case,
        "historical_half_frequency_rule_changes": changed_rows,
        "historical_half_frequency_rule_false_changes": false_changes,
        "audit_passed": audit_passed,
        "conclusion": conclusion,
    }
    with (OUTPUT_FOLDER / "fundamental_frequency_audit.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        fieldnames = list(rows[0].keys())
        for row in rows:
            for fieldname in row:
                if fieldname not in fieldnames:
                    fieldnames.append(fieldname)
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    (OUTPUT_FOLDER / "fundamental_frequency_audit.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].scatter(voltage, band_peak, s=18, alpha=0.8, label="200 declared-band peaks")
    voltage_line = np.linspace(voltage.min(), voltage.max(), 100)
    axes[0].plot(
        voltage_line,
        slope * voltage_line + intercept,
        color="black",
        linewidth=1.2,
        label=f"Linear trend (r = {correlation:.3f})",
    )
    axes[0].scatter(
        [harmonic_case["voltage_V"]], [harmonic_case["band_peak_Hz"]],
        color="crimson", s=45, zorder=3,
        label=(f"Sample {harmonic_case['sample_index']}: "
               f"{format_hz(harmonic_case['band_peak_Hz'])}"),
    )
    axes[0].set(
        xlabel="Motor voltage (V)", ylabel="Declared-band peak frequency (Hz)",
        title="Declared-band peak follows motor voltage",
    )
    axes[0].legend(fontsize=8)

    sample_index = harmonic_case["sample_index"]
    sample_frequency, sample_amplitude = spectrum_amplitude(acceleration[sample_index])
    display_band = (sample_frequency >= 20.0) & (sample_frequency <= 130.0)
    axes[1].plot(sample_frequency[display_band], sample_amplitude[display_band], linewidth=1.0)
    for frequency_hz, label, colour in (
        (harmonic_case["band_peak_Hz"], "Declared-band candidate", "forestgreen"),
        (harmonic_case["broad_peak_Hz"], f"Order-{harmonic_case['broad_peak_harmonic_order']} peak", "crimson"),
    ):
        axes[1].axvline(frequency_hz, color=colour, linestyle="--", linewidth=1)
        axes[1].annotate(
            f"{format_hz(frequency_hz)} {label}",
            (frequency_hz, sample_amplitude[int(np.argmin(np.abs(sample_frequency - frequency_hz)))]),
            xytext=(4, 8), textcoords="offset points", color=colour, fontsize=8,
        )
    axes[1].set(
        xlabel="Frequency (Hz)", ylabel="Amplitude (g)",
        title=f"Sample {sample_index}: automatically selected harmonic case",
    )
    figure.tight_layout()
    figure.savefig(OUTPUT_FOLDER / "fundamental_frequency_audit.png", dpi=220)
    plt.close(figure)

    print(json.dumps(summary, indent=2))
    if not audit_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
