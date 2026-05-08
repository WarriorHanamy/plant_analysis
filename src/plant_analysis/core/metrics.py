from __future__ import annotations

import control
import numpy as np

from plant_analysis.core.bode import freqresp_hz

__all__ = [
    "bandwidth_3db",
    "closed_loop_bandwidth",
    "open_loop_margin",
    "sensitivity_metrics",
    "system_performance",
]


def bandwidth_3db(
    system: control.TransferFunction, max_hz: float = 100.0, points: int = 8000
) -> float:
    """Find the first -3 dB bandwidth of a transfer function.

    Args:
        system: Continuous or discrete SISO transfer function.
        max_hz: Upper search frequency [Hz].
        points: Number of linearly spaced search points.

    Returns:
        First crossing frequency [Hz], or NaN if no crossing is found.
    """
    frequencies = np.linspace(1e-6, max_hz, points)
    mag_db, _ = freqresp_hz(system, frequencies)
    target = mag_db[0] - 3.0
    below = np.flatnonzero(mag_db <= target)
    if below.size == 0:
        return float("nan")
    idx = int(below[0])
    if idx == 0:
        return float(frequencies[0])
    return float(
        np.interp(target, [mag_db[idx - 1], mag_db[idx]], [frequencies[idx - 1], frequencies[idx]])
    )


def open_loop_margin(system: control.TransferFunction) -> tuple[float, float]:
    """Compute classical open-loop gain and phase margins.

    Args:
        system: Open-loop SISO transfer function L(s) or L(z).

    Returns:
        Gain margin [dB] and phase margin [deg]; unavailable margins are NaN.
    """
    gm, pm, _, _ = control.margin(system)
    gm_db = float(20.0 * np.log10(gm)) if np.isfinite(gm) and gm > 0 else float("nan")
    pm_deg = float(pm) if np.isfinite(pm) else float("nan")
    return gm_db, pm_deg


def closed_loop_bandwidth(open_loop: control.TransferFunction, max_hz: float = 100.0) -> float:
    """Compute -3 dB bandwidth of the unity-feedback closed-loop response.

    Args:
        open_loop: Open-loop SISO transfer function L.
        max_hz: Upper search frequency [Hz].

    Returns:
        Bandwidth [Hz] of T = L / (1 + L), or NaN if no crossing is found.
    """
    return bandwidth_3db(control.feedback(open_loop, 1), max_hz=max_hz)


def sensitivity_metrics(
    open_loop: control.TransferFunction, max_hz: float = 100.0
) -> dict[str, float]:
    """Evaluate unity-feedback sensitivity and complementary-sensitivity peaks.

    Args:
        open_loop: Open-loop SISO transfer function L.
        max_hz: Upper evaluation frequency [Hz].

    Returns:
        Dictionary with max S [dB], max T [dB], and closed-loop bandwidth [Hz].
    """
    frequencies = np.logspace(-2, np.log10(max_hz), 2000)
    sensitivity = 1 / (1 + open_loop)
    complementary = control.feedback(open_loop, 1)
    s_mag, _ = freqresp_hz(sensitivity, frequencies)
    t_mag, _ = freqresp_hz(complementary, frequencies)
    return {
        "max_sensitivity_db": float(np.max(s_mag)),
        "max_complementary_db": float(np.max(t_mag)),
        "closed_loop_bandwidth_hz": closed_loop_bandwidth(open_loop, max_hz=max_hz),
    }


def system_performance(
    open_loop: control.TransferFunction, max_hz: float = 100.0
) -> dict[str, float]:
    """Collect controller-tuning performance metrics for an open-loop model.

    Args:
        open_loop: Open-loop SISO transfer function L = controller * plant.
        max_hz: Upper evaluation frequency [Hz].

    Returns:
        Dictionary containing margins, sensitivity peaks, and closed-loop bandwidth.
    """
    gm_db, pm_deg = open_loop_margin(open_loop)
    metrics = sensitivity_metrics(open_loop, max_hz=max_hz)
    metrics.update({"gain_margin_db": gm_db, "phase_margin_deg": pm_deg})
    return metrics
