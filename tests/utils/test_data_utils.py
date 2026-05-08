import numpy as np

from plant_analysis.core.schemas import SweepResult, TimeSeries
from plant_analysis.utils.data_utils import load_npz, resample_irregular, save_npz


def make_series(time, values, name="signal"):
    return TimeSeries(time_s=time, values=values, signal_name=name)


def test_npz_round_trip(tmp_path):
    sweep = SweepResult(
        chirp=make_series([0, 1], [0, 1], "chirp"),
        reference=make_series([0, 1], [0, 1], "reference"),
        control=make_series([0, 1], [0, 1], "control"),
        output_filtered=make_series([0, 1], [0, 1], "output_filtered"),
        output_raw=make_series([0, 1], [0, 1], "output_raw"),
        fs_hz=1,
    )
    path = tmp_path / "sweep.npz"
    save_npz(path, sweep)
    loaded = load_npz(path, SweepResult)
    assert np.allclose(loaded.control.values, sweep.control.values)


def test_resample_irregular_returns_uniform_time():
    time, data, fs = resample_irregular(np.array([0.0, 0.1, 0.21]), np.array([0.0, 1.0, 0.0]))
    assert len(time) == len(data)
    assert fs > 0
