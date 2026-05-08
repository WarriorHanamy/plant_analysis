import numpy as np

from plant_analysis.core.bode import freqresp_hz
from plant_analysis.core.filters import butterworth_lowpass, notch_filter


def test_butterworth_lowpass_is_near_minus_3db_at_cutoff():
    system = butterworth_lowpass(2, 10)
    mag, _ = freqresp_hz(system, np.array([10.0]))
    assert np.isclose(mag[0], -3.0, atol=0.2)


def test_notch_filter_unit_gain_returns_identity():
    system = notch_filter(50, 0.5, 1.0)
    mag, _ = freqresp_hz(system, np.array([50.0]))
    assert np.isclose(mag[0], 0.0, atol=1e-9)
