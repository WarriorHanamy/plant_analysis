from __future__ import annotations

import control
import numpy as np
from scipy import signal as scipy_signal

from plant_analysis.core.schemas import PlantModel, SisoTransferFunction

__all__ = ["coherence", "fft_frd", "fit_discrete_transfer", "tfestimate", "to_transfer_function"]


def tfestimate(
    input_signal: np.ndarray, output_signal: np.ndarray, fs: float, nperseg: int | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Estimate frequency response with Welch cross/auto spectral densities.

    Args:
        input_signal: Excitation samples u[k] with shape (n,).
        output_signal: Response samples y[k] with shape (n,), aligned with input_signal.
        fs: Sampling frequency [Hz].
        nperseg: Welch segment length. Defaults to min(1024, n).

    Returns:
        Frequencies [Hz], complex FRD H(f) = P_yu(f) / P_uu(f), and coherence [0, 1].
    """
    input_signal = np.asarray(input_signal, dtype=float)
    output_signal = np.asarray(output_signal, dtype=float)
    if input_signal.shape != output_signal.shape:
        raise ValueError("input and output signals must have matching shape")
    if nperseg is None:
        nperseg = min(1024, len(input_signal))
    freq, pxy = scipy_signal.csd(output_signal, input_signal, fs=fs, nperseg=nperseg)
    _, pxx = scipy_signal.welch(input_signal, fs=fs, nperseg=nperseg)
    _, coh = scipy_signal.coherence(input_signal, output_signal, fs=fs, nperseg=nperseg)
    return freq, pxy / np.maximum(pxx, np.finfo(float).tiny), coh


def fft_frd(
    input_signal: np.ndarray, output_signal: np.ndarray, fs: float
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate frequency response by direct FFT division.

    Args:
        input_signal: Excitation samples u[k] with shape (n,).
        output_signal: Response samples y[k] with shape (n,), aligned with input_signal.
        fs: Sampling frequency [Hz].

    Returns:
        Frequencies [Hz] and complex FRD H(f) = FFT(y) / FFT(u).
    """
    input_signal = np.asarray(input_signal, dtype=float)
    output_signal = np.asarray(output_signal, dtype=float)
    freq = np.fft.rfftfreq(len(input_signal), d=1.0 / fs)
    u = np.fft.rfft(input_signal - np.mean(input_signal))
    y = np.fft.rfft(output_signal - np.mean(output_signal))
    return freq, y / np.maximum(u, np.finfo(float).tiny)


def fit_discrete_transfer(
    input_signal: np.ndarray,
    output_signal: np.ndarray,
    fs: float,
    na: int = 2,
    nb: int = 2,
    delay: int = 0,
) -> PlantModel:
    """Fit a discrete SISO ARX transfer function by least squares.

    The fitted model is y[k] + a1*y[k-1] + ... = b0*u[k-delay] + ... + b_nb*u[k-delay-nb].

    Args:
        input_signal: Excitation samples u[k] with shape (n,).
        output_signal: Response samples y[k] with shape (n,), aligned with input_signal.
        fs: Sampling frequency [Hz].
        na: Denominator order; result denominator shape is (na + 1,).
        nb: Numerator order; result numerator shape is (nb + 1,).
        delay: Integer input delay in samples.

    Returns:
        PlantModel containing a z^-1 SISO transfer function and identification metadata.
    """
    u = np.asarray(input_signal, dtype=float)
    y = np.asarray(output_signal, dtype=float)
    start = max(na, nb + delay)
    rows: list[list[float]] = []
    target: list[float] = []
    for k in range(start, len(y)):
        row = [-y[k - i] for i in range(1, na + 1)]
        row.extend(u[k - delay - j] for j in range(nb + 1))
        rows.append(row)
        target.append(y[k])
    theta, *_ = np.linalg.lstsq(np.asarray(rows), np.asarray(target), rcond=None)
    den = np.r_[1.0, theta[:na]]
    num = theta[na:]
    return PlantModel(
        transfer_function=SisoTransferFunction(
            numerator=num,
            denominator=den,
            domain_variable="z^-1",
            sample_time_s=1.0 / fs,
        ),
        method="least_squares_arx",
    )


def to_transfer_function(model: PlantModel) -> control.TransferFunction:
    """Convert a schema PlantModel into a python-control TransferFunction.

    Args:
        model: Plant schema with SISO numerator/denominator coefficients.

    Returns:
        Continuous or discrete python-control transfer function matching model.sample_time_s.
    """
    tf = model.transfer_function
    if tf.sample_time_s is None:
        return control.TransferFunction(tf.numerator, tf.denominator)
    return control.TransferFunction(tf.numerator, tf.denominator, tf.sample_time_s)


coherence = scipy_signal.coherence
