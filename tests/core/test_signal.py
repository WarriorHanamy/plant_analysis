import numpy as np
import pytest

from plant_analysis.core.signal import data_fft, kalman_filter


def test_data_fft_recovers_sine_frequency():
    fs = 100.0
    time = np.arange(0, 2, 1 / fs)
    data = np.sin(2 * np.pi * 5 * time)
    freq, mag, _, found_fs = data_fft(time, data)
    assert np.isclose(found_fs, fs)
    assert np.isclose(freq[np.argmax(mag)], 5.0, atol=0.1)


def test_kalman_filter_converges_to_constant_signal():
    filtered = kalman_filter(np.ones(100))
    assert filtered[-1] == pytest.approx(1.0, abs=1e-3)
