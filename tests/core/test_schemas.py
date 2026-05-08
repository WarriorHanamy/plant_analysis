import numpy as np
import pytest
from pydantic import ValidationError

from plant_analysis.core import schemas
from plant_analysis.core.schemas import (
    ControllerSynthesisInputPaths,
    ControllerSynthesisOutputPaths,
    DiagnosisOutputPaths,
    FilterResult,
    FilterSpec,
    PlantModel,
    SisoTransferFunction,
    SweepResult,
    TimeSeries,
    TuningTable,
)


def make_series(time, values, name="signal"):
    return TimeSeries(time_s=time, values=values, signal_name=name)


def test_sweep_result_rejects_mismatched_lengths():
    with pytest.raises(ValidationError):
        SweepResult(
            chirp=make_series([0, 1], [0, 1], "chirp"),
            reference=make_series([0, 1], [0, 1], "reference"),
            control=make_series([0], [0], "control"),
            output_filtered=make_series([0, 1], [0, 1], "output_filtered"),
            output_raw=make_series([0, 1], [0, 1], "output_raw"),
            fs_hz=100,
        )


def test_sweep_result_npz_payload_round_trip_shape():
    sweep = SweepResult(
        chirp=make_series(np.arange(3), np.ones(3), "chirp"),
        reference=make_series(np.arange(3), np.ones(3), "reference"),
        control=make_series(np.arange(3), np.ones(3), "control"),
        output_filtered=make_series(np.arange(3), np.ones(3), "output_filtered"),
        output_raw=make_series(np.arange(3), np.ones(3), "output_raw"),
        fs_hz=10,
    )
    assert sweep.as_npz_dict()["control"].shape == (3,)


def test_plant_model_exposes_transfer_function_shape():
    plant = PlantModel(
        transfer_function=SisoTransferFunction(
            numerator=[0.1, 0.2],
            denominator=[1.0, -0.7],
            domain_variable="z^-1",
            sample_time_s=0.01,
        ),
        input_signal="actuator",
        output_signal="gyro_rate",
        method="least_squares_arx",
    )
    assert plant.transfer_function.numerator.shape == (2,)
    assert plant.fs_hz == pytest.approx(100.0)


def test_pipeline_paths_default_to_input_and_output_dirs():
    input_paths = ControllerSynthesisInputPaths()
    output_paths = ControllerSynthesisOutputPaths()
    diagnosis_paths = DiagnosisOutputPaths()

    assert input_paths.recorded_csv.as_posix() == "_data_input/recorded_data.csv"
    assert output_paths.sweep_npz.as_posix() == "_output/sweep_result.npz"
    assert output_paths.plant_model_npz.as_posix() == "_output/plant_id.npz"
    assert output_paths.tuning_table_csv.as_posix() == "_output/tuning_table.csv"
    assert diagnosis_paths.filter_bode_png.as_posix() == "_output/figs/filter_bode.png"
    assert diagnosis_paths.filter_step_png.as_posix() == "_output/figs/filter_step.png"
    assert diagnosis_paths.filter_report_npz.as_posix() == "_output/filter_report.npz"


def test_schema_exports_only_real_array_shape_aliases():
    exported = set(schemas.__all__)

    assert "FloatArray1D" in exported
    assert "FloatMatrix2D" in exported
    assert "TimeVector" not in exported
    assert "SampleVector" not in exported
    assert "PolynomialCoefficients" not in exported


def test_filter_result_rejects_mismatched_response_shapes():
    with pytest.raises(ValidationError):
        FilterResult(
            numerator=[1.0],
            denominator=[1.0, 1.0],
            frequencies_hz=[1.0, 2.0],
            magnitude_db=[0.0],
            phase_deg=[0.0, -45.0],
        )


def test_filter_spec_validates_order():
    with pytest.raises(ValidationError):
        FilterSpec(filter_type="butterworth", order=9, cutoff=10)


def test_tuning_table_lengths_match():
    table = TuningTable(parameters=[{"Kp": 1.0}], metrics=[{"pm": 45.0}], status=["pass"])
    assert table.status == ["pass"]
