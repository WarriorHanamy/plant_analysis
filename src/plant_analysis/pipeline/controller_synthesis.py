from __future__ import annotations

import csv
from pathlib import Path

import control
import numpy as np

from plant_analysis.core.metrics import system_performance
from plant_analysis.core.schemas import (
    ControllerSynthesisOutputPaths,
    PlantModel,
    SweepResult,
    TuningConstraints,
    TuningTable,
)
from plant_analysis.core.sysid import fit_discrete_transfer, to_transfer_function
from plant_analysis.utils.data_utils import load_npz, save_npz


def identify(sweep: SweepResult, na: int = 2, nb: int = 2) -> PlantModel:
    plant = fit_discrete_transfer(
        sweep.control.values, sweep.output_filtered.values, sweep.fs_hz, na=na, nb=nb
    )
    return PlantModel(
        transfer_function=plant.transfer_function,
        input_signal=sweep.control.signal_name,
        output_signal=sweep.output_filtered.signal_name,
        method=plant.method,
    )


def _grid(spec: tuple[float, float, int]) -> np.ndarray:
    start, stop, count = spec
    if count <= 1:
        return np.asarray([start], dtype=float)
    return np.linspace(start, stop, int(count))


def tune(plant: PlantModel, constraints: TuningConstraints | None = None) -> TuningTable:
    constraints = constraints or TuningConstraints()
    system = to_transfer_function(plant)
    parameters: list[dict[str, float]] = []
    metrics: list[dict[str, float]] = []
    status: list[str] = []
    for kp in _grid(constraints.kp):
        for ki in _grid(constraints.ki):
            for kd in _grid(constraints.kd):
                controller = control.TransferFunction([kd, kp, ki], [1.0, 0.0], system.dt)
                perf = system_performance(
                    controller * system, max_hz=(plant.fs_hz or 200.0) / 2.0 * 0.9
                )
                ok = (
                    np.isnan(perf["phase_margin_deg"])
                    or perf["phase_margin_deg"] >= constraints.min_phase_margin_deg
                ) and perf["max_sensitivity_db"] <= constraints.max_sensitivity_db
                parameters.append({"Kp": float(kp), "Ki": float(ki), "Kd": float(kd)})
                metrics.append(perf)
                status.append("pass" if ok else "fail")
    return TuningTable(parameters=parameters, metrics=metrics, status=status)


def write_tuning_csv(path: str | Path, table: TuningTable) -> None:
    fields = sorted({key for row in table.parameters + table.metrics for key in row}) + ["status"]
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for params, metric, state in zip(
            table.parameters, table.metrics, table.status, strict=True
        ):
            writer.writerow({**params, **metric, "status": state})


def run(paths: ControllerSynthesisOutputPaths | None = None) -> TuningTable:
    paths = paths or ControllerSynthesisOutputPaths()
    sweep = load_npz(paths.sweep_npz, SweepResult)
    plant = identify(sweep)
    table = tune(plant)
    paths.output_dir.mkdir(parents=True, exist_ok=True)
    save_npz(paths.plant_model_npz, plant)
    write_tuning_csv(paths.tuning_table_csv, table)
    return table
