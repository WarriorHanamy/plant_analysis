from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy import interpolate

from plant_analysis.core.schemas import (
    NumericModel,
    PlantModel,
    SisoTransferFunction,
    SweepResult,
    TimeSeries,
)


def read_csv_sweep(path: str | Path) -> SweepResult:
    data = np.genfromtxt(path, delimiter=",", names=True, dtype=float, deletechars="")
    names = set(data.dtype.names or [])
    required = {"sweep_time", "chirp", "r", "u", "y_flt", "y_raw"}
    if not required.issubset(names):
        aliases = {
            "time": "sweep_time",
            "r": "r",
            "u": "u",
            "input": "u",
            "output": "y_raw",
            "y": "y_raw",
            "filtered": "y_flt",
            "reference": "r",
            "sweep_chirp": "chirp",
        }
        values = {target: data[source] for source, target in aliases.items() if source in names}
        values.setdefault("chirp", values.get("u"))
        values.setdefault("y_flt", values.get("y_raw"))
    else:
        values = {name: data[name] for name in required}
    missing = required - set(values)
    if missing:
        raise ValueError(f"missing sweep CSV columns: {sorted(missing)}")
    time = np.asarray(values["sweep_time"], dtype=float)
    fs = 1.0 / float(np.median(np.diff(time)))
    return SweepResult(
        chirp=TimeSeries(time_s=time, values=values["chirp"], signal_name="chirp"),
        reference=TimeSeries(time_s=time, values=values["r"], signal_name="reference"),
        control=TimeSeries(time_s=time, values=values["u"], signal_name="control"),
        output_filtered=TimeSeries(
            time_s=time, values=values["y_flt"], signal_name="output_filtered"
        ),
        output_raw=TimeSeries(time_s=time, values=values["y_raw"], signal_name="output_raw"),
        fs_hz=fs,
    )


def save_npz(path: str | Path, model: NumericModel | dict[str, np.ndarray | float | str]) -> None:
    payload = model.as_npz_dict() if isinstance(model, NumericModel) else model
    np.savez(path, **payload)


def load_npz[T: NumericModel](
    path: str | Path, schema: type[T] | None = None
) -> T | dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        payload = {key: data[key] for key in data.files}
    if schema is None:
        return payload
    if schema is SweepResult:
        time = payload["time_s"]
        return schema(
            chirp=TimeSeries(time_s=time, values=payload["chirp"], signal_name="chirp"),
            reference=TimeSeries(time_s=time, values=payload["reference"], signal_name="reference"),
            control=TimeSeries(time_s=time, values=payload["control"], signal_name="control"),
            output_filtered=TimeSeries(
                time_s=time, values=payload["output_filtered"], signal_name="output_filtered"
            ),
            output_raw=TimeSeries(
                time_s=time, values=payload["output_raw"], signal_name="output_raw"
            ),
            fs_hz=float(payload["fs_hz"]),
        )
    if schema is PlantModel:
        sample_time = float(payload["sample_time_s"])
        return schema(
            transfer_function=SisoTransferFunction(
                numerator=payload["numerator"],
                denominator=payload["denominator"],
                domain_variable=str(payload["domain_variable"]),
                sample_time_s=None if np.isnan(sample_time) else sample_time,
            ),
            input_signal=str(payload["input_signal"]),
            output_signal=str(payload["output_signal"]),
            method=str(payload["method"]),
        )
    return schema(**payload)


def resample_irregular(
    time: np.ndarray, data: np.ndarray, fs: float | None = None
) -> tuple[np.ndarray, np.ndarray, float]:
    time = np.asarray(time, dtype=float)
    data = np.asarray(data, dtype=float)
    if fs is None:
        fs = 1.0 / float(np.median(np.diff(np.sort(time))))
    ts = 1.0 / fs
    new_time = np.arange(time[0], time[-1] + ts * 0.5, ts)
    new_data = interpolate.interp1d(
        time, data, axis=0, bounds_error=False, fill_value="extrapolate"
    )(new_time)
    return new_time, new_data, fs
