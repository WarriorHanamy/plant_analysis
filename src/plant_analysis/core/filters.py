from __future__ import annotations

import control
import numpy as np

from plant_analysis.utils.filter_utils import butterworth_coefficients, prewarp_frequency

__all__ = [
    "boost",
    "butter_lowpass",
    "butterworth_lowpass",
    "differentiator",
    "notch_filter",
    "pi_lead",
    "pi_leadlag",
]


def _tf(num: list[float] | np.ndarray, den: list[float] | np.ndarray) -> control.TransferFunction:
    return control.TransferFunction(np.asarray(num, dtype=float), np.asarray(den, dtype=float))


def _discretize(system: control.TransferFunction, fs: float | None) -> control.TransferFunction:
    if fs is None:
        return system
    if fs <= 0:
        raise ValueError("fs must be positive")
    return control.sample_system(system, 1.0 / fs, method="tustin")


def butter_lowpass(order: int, cutoff: float, fs: float | None = None) -> control.TransferFunction:
    """Create a normalized Butterworth low-pass transfer function.

    Args:
        order: Number of low-pass poles.
        cutoff: Cutoff frequency, unit [Hz]. Prewarped before discretization when fs is set.
        fs: Sampling frequency [Hz]. If None, return a continuous-time model.

    Returns:
        SISO low-pass transfer function with -3 dB gain near cutoff.
    """
    fc = prewarp_frequency(cutoff, fs) if fs is not None else cutoff
    denominator = butterworth_coefficients(order, fc)
    system = _tf([1.0], denominator)
    return _discretize(system, fs)


def butterworth_lowpass(
    order: int,
    cutoff: float,
    fs: float | None = None,
    zero_order: int = 0,
    zero_cutoff: float | None = None,
) -> control.TransferFunction:
    """Create a Butterworth pole-zero low-pass or lead-like transfer function.

    Args:
        order: Number of denominator poles.
        cutoff: Denominator cutoff frequency [Hz].
        fs: Sampling frequency [Hz]. If None, return a continuous-time model.
        zero_order: Number of numerator zeros represented by a Butterworth polynomial.
        zero_cutoff: Numerator cutoff frequency [Hz]. Defaults to cutoff.

    Returns:
        SISO transfer function whose numerator and denominator are Butterworth polynomials.
    """
    fp = prewarp_frequency(cutoff, fs) if fs is not None else cutoff
    fz = zero_cutoff if zero_cutoff is not None else cutoff
    fz = prewarp_frequency(fz, fs) if fs is not None else fz
    numerator = butterworth_coefficients(zero_order, fz)
    denominator = butterworth_coefficients(order, fp)
    system = _tf(numerator, denominator)
    return _discretize(system, fs)


def notch_filter(
    fc: float, bandwidth: float, gain: float, fs: float | None = None
) -> control.TransferFunction:
    """Create a second-order notch or anti-notch filter.

    Args:
        fc: Notch center frequency [Hz].
        bandwidth: Damping-like bandwidth ratio used by the MATLAB source.
        gain: Depth/boost ratio; 1.0 returns an identity transfer function.
        fs: Sampling frequency [Hz]. If None, return a continuous-time model.

    Returns:
        SISO transfer function with numerator damping bandwidth and denominator bandwidth/gain.
    """
    if np.isclose(gain, 1.0):
        return _discretize(_tf([1.0], [1.0]), fs)
    fd = prewarp_frequency(fc, fs) if fs is not None else fc
    wn = 2.0 * np.pi * fd
    numerator = [1.0, 2.0 * bandwidth * wn, wn**2]
    denominator = [1.0, 2.0 * (bandwidth / gain) * wn, wn**2]
    return _discretize(_tf(numerator, denominator), fs)


def pi_lead(
    kp: float, fi: float, lead_fc: float, lead_gain: float, fs: float | None = None
) -> control.TransferFunction:
    """Create a PI controller multiplied by one lead compensator.

    Args:
        kp: Proportional gain.
        fi: Integral zero frequency in the PI term, unit [rad/s] as in Kp * (1 + fi / s).
        lead_fc: Lead corner frequency [Hz].
        lead_gain: Lead separation ratio; 1.0 disables the lead term.
        fs: Sampling frequency [Hz]. If None, return a continuous-time controller.

    Returns:
        SISO controller transfer function.
    """
    pi = _tf([kp, kp * fi], [1.0, 0.0])
    if np.isclose(lead_gain, 1.0):
        lead = _tf([1.0], [1.0])
    else:
        lead = _tf(
            [1.0 / (2.0 * np.pi * lead_fc), 1.0], [1.0 / (2.0 * np.pi * lead_fc * lead_gain), 1.0]
        )
    return _discretize(lead * pi, fs)


def pi_leadlag(
    kp: float,
    fi: float,
    lead_fc: float,
    lead_gain: float,
    lag_fc: float,
    lag_gain: float,
    fs: float | None = None,
) -> control.TransferFunction:
    """Create a PI controller with one lead and one lag compensator.

    Args:
        kp: Proportional gain.
        fi: Integral zero frequency in the PI term, unit [rad/s].
        lead_fc: Lead corner frequency [Hz].
        lead_gain: Lead separation ratio.
        lag_fc: Lag corner frequency [Hz].
        lag_gain: Lag separation ratio.
        fs: Sampling frequency [Hz]. If None, return a continuous-time controller.

    Returns:
        SISO controller transfer function.
    """
    lead = pi_lead(kp, fi, lead_fc, lead_gain, None)
    lag = _tf([1.0 / (2.0 * np.pi * lag_fc * lag_gain), 1.0], [1.0 / (2.0 * np.pi * lag_fc), 1.0])
    return _discretize(lead * lag, fs)


def boost(boost_fc: float, boost_gain: float, fs: float | None = None) -> control.TransferFunction:
    """Create a first-order high-frequency boost compensator.

    Args:
        boost_fc: Boost corner frequency [Hz].
        boost_gain: High-frequency gain ratio; 1.0 returns identity.
        fs: Sampling frequency [Hz]. If None, return a continuous-time model.

    Returns:
        SISO boost transfer function.
    """
    if np.isclose(boost_gain, 1.0):
        return _discretize(_tf([1.0], [1.0]), fs)
    system = (
        _tf(
            [1.0 / (2.0 * np.pi * boost_fc), 1.0],
            [1.0 / (2.0 * np.pi * boost_fc / boost_gain), 1.0],
        )
        * boost_gain
    )
    return _discretize(system, fs)


def differentiator(order: int, cutoff: float, fs: float | None = None) -> control.TransferFunction:
    """Create a filtered differentiator s / Butterworth(s).

    Args:
        order: Number of low-pass poles that limit differentiator noise gain.
        cutoff: Low-pass cutoff frequency [Hz].
        fs: Sampling frequency [Hz]. If None, return a continuous-time model.

    Returns:
        SISO differentiator transfer function.
    """
    fc = prewarp_frequency(cutoff, fs) if fs is not None else cutoff
    denominator = butterworth_coefficients(order, fc)
    return _discretize(_tf([1.0, 0.0], denominator), fs)
