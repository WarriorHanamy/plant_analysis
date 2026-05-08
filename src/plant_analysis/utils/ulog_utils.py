from __future__ import annotations

from pathlib import Path

import numpy as np

from plant_analysis.core.schemas import FilterChainReport, SweepResult, TimeSeries


def _ulog(path: str | Path):
    try:
        from pyulog import ULog
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("pyulog is required to read ULog files") from exc
    return ULog(str(path))


def read_ulog_filter_chain(path: str | Path) -> FilterChainReport:
    ulog = _ulog(path)
    signal_names: list[str] = []
    fft_peaks: dict[str, float] = {}
    for dataset in ulog.data_list:
        for name, values in dataset.data.items():
            if "gyro" not in name.lower() and "rate" not in name.lower():
                continue
            arr = np.asarray(values, dtype=float)
            if arr.size < 2:
                continue
            signal = f"{dataset.name}.{name}"
            signal_names.append(signal)
            spectrum = np.abs(np.fft.rfft(arr - np.mean(arr)))
            fft_peaks[signal] = float(np.max(spectrum))
    return FilterChainReport(signal_names=signal_names, fft_peaks=fft_peaks)


def read_ulog_chirp(path: str | Path) -> SweepResult:
    ulog = _ulog(path)
    for dataset in ulog.data_list:
        names = set(dataset.data)
        if {"timestamp", "chirp", "u", "y"}.issubset(names):
            time = np.asarray(dataset.data["timestamp"], dtype=float) * 1e-6
            time = time - time[0]
            fs = 1.0 / float(np.median(np.diff(time)))
            y = np.asarray(dataset.data["y"], dtype=float)
            return SweepResult(
                chirp=TimeSeries(time_s=time, values=dataset.data["chirp"], signal_name="chirp"),
                reference=TimeSeries(
                    time_s=time,
                    values=dataset.data.get("r", dataset.data["chirp"]),
                    signal_name="reference",
                ),
                control=TimeSeries(time_s=time, values=dataset.data["u"], signal_name="control"),
                output_filtered=TimeSeries(
                    time_s=time, values=dataset.data.get("y_flt", y), signal_name="output_filtered"
                ),
                output_raw=TimeSeries(time_s=time, values=y, signal_name="output_raw"),
                fs_hz=fs,
            )
    raise ValueError("no chirp-like dataset found in ULog")
