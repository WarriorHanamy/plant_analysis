from __future__ import annotations

import control
import numpy as np
from scipy import interpolate

from plant_analysis.core.bode import freqresp_hz

__all__ = ["data_fft", "kalman_filter", "time_delay_vector"]


def data_fft(
    time: np.ndarray, data: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Compute a single-sided FFT magnitude from an irregular scalar time series.

    Args:
        time: Sample times with shape (n,), unit [s].
        data: Scalar samples with shape (n,), one value at each time[k].

    Returns:
        Frequencies [Hz], single-sided magnitude, de-meaned uniformly sampled data, and fs [Hz].
    """
    time = np.asarray(time, dtype=float)
    data = np.asarray(data, dtype=float)
    if time.ndim != 1 or data.ndim != 1 or len(time) != len(data):
        raise ValueError("time and data must be one-dimensional arrays with matching length")
    ts = float(np.median(np.diff(np.sort(time))))
    fs = 1.0 / ts
    interp_time = np.arange(time[0], time[-1] + ts * 0.5, ts)
    interp_data = interpolate.interp1d(time, data, bounds_error=False, fill_value="extrapolate")(
        interp_time
    )
    interp_data = interp_data - np.mean(interp_data)
    fft = np.fft.rfft(interp_data)
    mag = np.abs(fft / len(interp_data))
    if len(mag) > 2:
        mag[1:-1] *= 2.0
    freq = np.fft.rfftfreq(len(interp_data), d=ts)
    return freq, mag, interp_data, fs


def kalman_filter(
    z: np.ndarray, q: float = 1e-5, r: float = 1e-2, x0: float | None = None, p0: float = 1.0
) -> np.ndarray:
    """Apply a scalar random-walk Kalman filter to a measurement sequence.

    Args:
        z: Measurement samples with shape (n,).
        q: Process noise variance per sample.
        r: Measurement noise variance per sample.
        x0: Initial state estimate. Defaults to z[0].
        p0: Initial state covariance.

    Returns:
        Filtered state estimate with shape (n,).
    """
    z = np.asarray(z, dtype=float)
    x = float(z[0] if x0 is None else x0)
    p = float(p0)
    out = np.empty_like(z)
    for idx, measurement in enumerate(z):
        p += q
        k = p / (p + r)
        x += k * (measurement - x)
        p = (1.0 - k) * p
        out[idx] = x
    return out


def time_delay_vector(
    system: control.TransferFunction, min_hz: float = 0.01, max_hz: float = 50.0, points: int = 1000
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate equivalent phase delay over a frequency range.

    Args:
        system: Continuous or discrete SISO transfer function.
        min_hz: Lower evaluation frequency [Hz].
        max_hz: Upper evaluation frequency [Hz].
        points: Number of linearly spaced evaluation points.

    Returns:
        Frequencies [Hz] and equivalent delay [s] computed from -phase / frequency.
    """
    frequencies = np.linspace(min_hz, max_hz, points)
    _, phase = freqresp_hz(system, frequencies)
    delay = -phase / 360.0 / frequencies
    return frequencies, delay
