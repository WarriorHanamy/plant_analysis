from __future__ import annotations

import numpy as np

from plant_analysis.core.schemas import (
    ControllerSynthesisInputPaths,
    ControllerSynthesisOutputPaths,
    SweepResult,
    TimeSeries,
)
from plant_analysis.utils.data_utils import read_csv_sweep, save_npz


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
    return sweep
