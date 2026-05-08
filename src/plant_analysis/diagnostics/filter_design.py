from __future__ import annotations

import numpy as np

from plant_analysis.core.bode import freqresp_hz
from plant_analysis.core.filters import butterworth_lowpass, differentiator, notch_filter
from plant_analysis.core.schemas import DiagnosisOutputPaths, FilterResult, FilterSpec
from plant_analysis.utils.plot_utils import save_bode_plot


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
    return result
