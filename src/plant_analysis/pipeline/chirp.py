from __future__ import annotations

import numpy as np

from plant_analysis.core.schemas import (
    ControllerSynthesisInputPaths,
    ControllerSynthesisOutputPaths,
    SweepResult,
    TimeSeries,
)
from plant_analysis.utils.data_utils import read_csv_sweep, save_npz
from plant_analysis.utils.plot_utils import (
    save_fft_magnitude_plot,
    save_spectrogram,
    save_time_plot,
)


def _chirp_fft(values: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    n = len(values)
    freq = np.fft.rfftfreq(n, d=1.0 / fs)
    mag = 20.0 * np.log10(np.maximum(np.abs(np.fft.rfft(values - np.mean(values))), 1e-15))
    return freq, mag


def run(
    input_paths: ControllerSynthesisInputPaths | None = None,
    output_paths: ControllerSynthesisOutputPaths | None = None,
) -> SweepResult:
    input_paths = input_paths or ControllerSynthesisInputPaths()
    output_paths = output_paths or ControllerSynthesisOutputPaths()
    sweep = read_csv_sweep(input_paths.recorded_csv)
    time = sweep.time_s - sweep.time_s[0]
    sweep = SweepResult(
        chirp=TimeSeries(
            time_s=time,
            values=sweep.chirp.values - np.mean(sweep.chirp.values),
            signal_name=sweep.chirp.signal_name,
            unit=sweep.chirp.unit,
        ),
        reference=TimeSeries(
            time_s=time,
            values=sweep.reference.values - np.mean(sweep.reference.values),
            signal_name=sweep.reference.signal_name,
            unit=sweep.reference.unit,
        ),
        control=TimeSeries(
            time_s=time,
            values=sweep.control.values - np.mean(sweep.control.values),
            signal_name=sweep.control.signal_name,
            unit=sweep.control.unit,
        ),
        output_filtered=TimeSeries(
            time_s=time,
            values=sweep.output_filtered.values - np.mean(sweep.output_filtered.values),
            signal_name=sweep.output_filtered.signal_name,
            unit=sweep.output_filtered.unit,
        ),
        output_raw=TimeSeries(
            time_s=time,
            values=sweep.output_raw.values - np.mean(sweep.output_raw.values),
            signal_name=sweep.output_raw.signal_name,
            unit=sweep.output_raw.unit,
        ),
        fs_hz=sweep.fs_hz,
    )
    output_paths.sweep_npz.parent.mkdir(parents=True, exist_ok=True)
    save_npz(output_paths.sweep_npz, sweep)

    figs = output_paths.chirp_figs_dir
    figs.mkdir(parents=True, exist_ok=True)

    # 1.1 Time-domain all signals
    save_time_plot(
        figs / "chirp_time_domain.png",
        sweep.time_s,
        [
            ("chirp", sweep.chirp.values),
            ("reference", sweep.reference.values),
            ("control", sweep.control.values),
            ("output_filtered", sweep.output_filtered.values),
            ("output_raw", sweep.output_raw.values),
        ],
    )

    # 1.2 Spectrogram of chirp excitation
    save_spectrogram(
        figs / "chirp_spectrogram.png",
        sweep.chirp.values,
        sweep.fs_hz,
        title="Chirp Excitation Spectrogram",
    )

    # 1.3 FFT magnitude of chirp
    freq, mag = _chirp_fft(sweep.chirp.values, sweep.fs_hz)
    save_fft_magnitude_plot(figs / "chirp_excitation_fft.png", freq, mag, label="chirp")

    # 1.4 Control vs output (raw + filtered)
    save_time_plot(
        figs / "chirp_control_vs_output.png",
        sweep.time_s,
        [
            ("control", sweep.control.values),
            ("output_filtered", sweep.output_filtered.values),
            ("output_raw", sweep.output_raw.values),
        ],
    )

    # 1.5 Sampling regularity
    dt = np.diff(sweep.time_s)
    dt_idx = np.arange(len(dt))
    save_time_plot(
        figs / "chirp_sampling_regularity.png",
        dt_idx / sweep.fs_hz,
        [("dt", dt * 1000)],
        ylabel="dt [ms]",
    )

    return sweep
