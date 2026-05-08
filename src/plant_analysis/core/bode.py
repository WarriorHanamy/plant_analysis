from __future__ import annotations

import control
import numpy as np

__all__ = ["freqresp", "freqresp_hz"]


def freqresp(system: control.TransferFunction, omega: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate a SISO transfer function on angular frequencies.

    Args:
        system: Continuous or discrete SISO transfer function.
        omega: Angular frequencies with shape (n,), unit [rad/s].

    Returns:
        Tuple of magnitude [dB] and unwrapped phase [deg], both shape (n,).
    """
    omega = np.asarray(omega, dtype=float)
    response = np.atleast_1d(control.frequency_response(system, omega).frdata.squeeze())
    magnitude_db = 20.0 * np.log10(np.maximum(np.abs(response), np.finfo(float).tiny))
    phase_deg = np.unwrap(np.angle(response)) * 180.0 / np.pi
    return magnitude_db, phase_deg


def freqresp_hz(
    system: control.TransferFunction, frequencies_hz: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate a SISO transfer function on ordinary frequencies.

    Args:
        system: Continuous or discrete SISO transfer function.
        frequencies_hz: Frequencies with shape (n,), unit [Hz].

    Returns:
        Tuple of magnitude [dB] and unwrapped phase [deg], both shape (n,).
    """
    return freqresp(system, 2.0 * np.pi * np.asarray(frequencies_hz, dtype=float))
