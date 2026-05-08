import numpy as np

from plant_analysis.utils.filter_utils import butterworth_coefficients, prewarp_frequency


def test_prewarp_frequency_matches_matlab_formula():
    assert np.isclose(prewarp_frequency(50, 200), (200 / np.pi) * np.tan(np.pi * 50 / 200))


def test_butterworth_coefficients_order_two():
    coef = butterworth_coefficients(2, 10)
    assert coef.shape == (3,)
    assert coef[-1] == 1.0
