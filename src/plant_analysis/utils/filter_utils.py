from __future__ import annotations

import importlib

import numpy as np

_BUTTERWORTH_COEFFICIENTS: dict[int, list[float]] = {
    0: [1.0],
    1: [1.0, 1.0],
    2: [1.0, 1.414213562373095, 1.0],
    3: [1.0, 2.0, 2.0, 1.0],
    4: [1.0, 2.613125929752753, 3.414213562373095, 2.613125929752753, 1.0],
    5: [1.0, 3.236067977499789, 5.236067977499789, 5.236067977499789, 3.236067977499789, 1.0],
    6: [
        1.0,
        3.863703305156273,
        7.464101615137753,
        9.141620172685640,
        7.464101615137753,
        3.863703305156273,
        1.0,
    ],
    7: [
        1.0,
        4.493959207434933,
        10.097834679044610,
        14.591793886479543,
        14.591793886479543,
        10.097834679044610,
        4.493959207434933,
        1.0,
    ],
    8: [
        1.0,
        5.125830895483012,
        13.137071184544089,
        21.846150969207624,
        25.688355931461274,
        21.846150969207628,
        13.137071184544089,
        5.125830895483013,
        1.0,
    ],
}


def prewarp_frequency(fc: float, fs: float) -> float:
    if fc <= 0 or fs <= 0:
        raise ValueError("fc and fs must be positive")
    if fc >= fs / 2:
        raise ValueError("fc must be below Nyquist frequency")
    return float((fs / np.pi) * np.tan(np.pi * fc / fs))


def butterworth_coefficients(order: int, fc: float) -> np.ndarray:
    if order not in _BUTTERWORTH_COEFFICIENTS:
        raise ValueError("order must be within [0, 8]")
    if fc <= 0:
        raise ValueError("fc must be positive")
    coef = np.asarray(_BUTTERWORTH_COEFFICIENTS[order], dtype=float)
    scale = 2.0 * np.pi * fc
    for idx in range(order):
        coef[idx] /= scale ** (order - idx)
    return coef


def solve_cutoff_by_order(
    order: int, target_hz: float, fs: float | None = None, has_zeros: bool = False
) -> float:
    if target_hz <= 0:
        raise ValueError("target_hz must be positive")
    high = (fs / 2 * 0.999) if fs is not None else target_hz * 100.0
    low = 1e-9
    filters = importlib.import_module("plant_analysis.core.filters")
    metrics = importlib.import_module("plant_analysis.core.metrics")

    for _ in range(80):
        mid = (low + high) / 2.0
        system = filters.butterworth_lowpass(order, mid, fs=fs)
        bw = metrics.bandwidth_3db(system, max_hz=max(target_hz * 4.0, mid * 4.0))
        if np.isnan(bw) or bw > target_hz:
            high = mid
        else:
            low = mid
    solved = (low + high) / 2.0
    if has_zeros:
        return solved
    return solved


f_predist_fc = prewarp_frequency
f_get_buterworth_coef = butterworth_coefficients
