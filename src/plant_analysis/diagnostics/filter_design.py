from __future__ import annotations

import numpy as np

from plant_analysis.core.bode import freqresp_hz
from plant_analysis.core.filters import butterworth_lowpass, differentiator, notch_filter
from plant_analysis.core.schemas import DiagnosisOutputPaths, FilterResult, FilterSpec
from plant_analysis.utils.plot_utils import (
    save_bode_plot,
    save_impulse_response_plot,
    save_pole_zero_plot,
    save_step_response_plot,
    save_time_plot,
)


def synthesize(spec: FilterSpec):
    if spec.filter_type == "butterworth":
        return butterworth_lowpass(spec.order, spec.cutoff, fs=spec.fs)
    if spec.filter_type == "notch":
        return notch_filter(spec.cutoff, spec.bandwidth or 0.5, spec.gain, fs=spec.fs)
    if spec.filter_type == "differentiator":
        return differentiator(spec.order, spec.cutoff, fs=spec.fs)
    raise ValueError(f"filter type {spec.filter_type!r} is not supported by filter_design")


def run(spec: FilterSpec, paths: DiagnosisOutputPaths | None = None) -> FilterResult:
    paths = paths or DiagnosisOutputPaths()
    system = synthesize(spec)
    max_hz = (spec.fs / 2.0 * 0.95) if spec.fs else spec.cutoff * 20.0
    frequencies = np.logspace(np.log10(max(spec.cutoff / 100.0, 1e-3)), np.log10(max_hz), 800)
    magnitude, phase = freqresp_hz(system, frequencies)
    result = FilterResult(
        numerator=np.asarray(system.num[0][0], dtype=float),
        denominator=np.asarray(system.den[0][0], dtype=float),
        frequencies_hz=frequencies,
        magnitude_db=magnitude,
        phase_deg=phase,
    )
    paths.figs_dir.mkdir(parents=True, exist_ok=True)
    save_bode_plot(paths.filter_bode_png, frequencies, magnitude, phase)

    # 5.2 Step response
    save_step_response_plot(paths.filter_step_png, system)

    # 5.3 Impulse response
    save_impulse_response_plot(paths.filter_impulse_png, system)

    # 5.4 Pole-zero map
    save_pole_zero_plot(paths.filter_pole_zero_png, system)

    # 5.5 Group delay
    phase_rad = np.unwrap(np.angle(np.atleast_1d(system(2j * np.pi * frequencies))))
    group_delay = -np.gradient(phase_rad, 2.0 * np.pi * frequencies)
    save_time_plot(
        paths.filter_group_delay_png,
        frequencies,
        [("group delay", group_delay)],
        ylabel="Group Delay [s]",
        xlabel="Frequency [Hz]",
    )

    # 5.6 Phase delay
    phase_delay = np.where(
        np.abs(frequencies) > 1e-10, -phase_rad / (2.0 * np.pi * frequencies), group_delay[0]
    )
    save_time_plot(
        paths.filter_phase_delay_png,
        frequencies,
        [("phase delay", phase_delay)],
        ylabel="Phase Delay [s]",
        xlabel="Frequency [Hz]",
    )

    # 5.7 Chirp-like time sweep (filtered vs unfiltered)
    t_end = 1.0 / max(spec.cutoff * 0.1, 1.0)
    import control as _ctrl

    if _ctrl.isdtime(system, strict=True):
        dt_val = system.dt if system.dt and system.dt is not True else 0.01
        t = np.arange(0, t_end + dt_val / 2, dt_val)
    else:
        t = np.linspace(0, t_end, 1000)
    sweep_input = np.sin(2.0 * np.pi * spec.cutoff * t * np.linspace(0, 1, len(t)))
    resp = system.forced_response(T=t, U=sweep_input, X0=0.0)
    save_time_plot(
        paths.filter_time_sweep_png,
        np.asarray(resp.time).ravel(),
        [("input", sweep_input), ("output", np.asarray(resp.outputs).ravel())],
    )

    # 5.8 Cascade response (filter applied twice)
    cascade = system * system
    mag_c, phase_c = freqresp_hz(cascade, frequencies)
    save_bode_plot(paths.filter_cascade_png, frequencies, mag_c, phase_c)

    return result
