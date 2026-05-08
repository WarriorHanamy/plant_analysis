from plant_analysis.core.filters import butterworth_lowpass
from plant_analysis.core.metrics import bandwidth_3db, open_loop_margin


def test_bandwidth_3db_known_lowpass():
    assert abs(bandwidth_3db(butterworth_lowpass(2, 10), max_hz=40) - 10) < 0.5


def test_open_loop_margin_returns_numbers():
    gm, pm = open_loop_margin(butterworth_lowpass(1, 10))
    assert isinstance(gm, float)
    assert isinstance(pm, float)
