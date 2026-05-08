import numpy as np

from plant_analysis.core.schemas import (
    ControllerSynthesisInputPaths,
    ControllerSynthesisOutputPaths,
    SweepResult,
    TimeSeries,
    TuningConstraints,
)
from plant_analysis.pipeline.chirp import run as run_chirp
from plant_analysis.pipeline.controller_synthesis import identify, tune
from plant_analysis.pipeline.controller_synthesis import run as run_synthesis

RECORDED_INPUT_DIR = "_data_input"


def test_identify_and_tune_end_to_end():
    rng = np.random.default_rng(2)
    u = rng.normal(size=200)
    y = np.zeros_like(u)
    for k in range(1, len(u)):
        y[k] = 0.7 * y[k - 1] + 0.1 * u[k]
    time = np.arange(len(u)) / 100
    sweep = SweepResult(
        chirp=TimeSeries(time_s=time, values=u, signal_name="chirp"),
        reference=TimeSeries(time_s=time, values=u, signal_name="reference"),
        control=TimeSeries(time_s=time, values=u, signal_name="control"),
        output_filtered=TimeSeries(time_s=time, values=y, signal_name="output_filtered"),
        output_raw=TimeSeries(time_s=time, values=y, signal_name="output_raw"),
        fs_hz=100,
    )
    plant = identify(sweep, na=1, nb=0)
    table = tune(plant, TuningConstraints(kp=(0.1, 0.2, 2), ki=(0.0, 0.0, 1), kd=(0.0, 0.0, 1)))
    assert len(table.parameters) == 2


def test_recorded_x_axis_csv_runs_chirp_and_synthesis_e2e(tmp_path):
    input_paths = ControllerSynthesisInputPaths(input_dir=RECORDED_INPUT_DIR)
    output_paths = ControllerSynthesisOutputPaths(output_dir=tmp_path)
    sweep = run_chirp(input_paths, output_paths)

    assert input_paths.recorded_csv.name == "recorded_data.csv"
    assert output_paths.sweep_npz.exists()
    assert len(sweep.time_s) > 10
    assert sweep.fs_hz > 0
    assert sweep.control.values.shape == sweep.output_filtered.values.shape

    table = run_synthesis(output_paths)

    assert table.parameters
    assert len(table.parameters) == len(table.metrics) == len(table.status)
    assert output_paths.plant_model_npz.exists()
    assert output_paths.tuning_table_csv.exists()
