import numpy as np

from plant_analysis.core.sysid import fit_discrete_transfer, tfestimate


def test_tfest_identifies_first_order_arx_shape():
    rng = np.random.default_rng(1)
    u = rng.normal(size=500)
    y = np.zeros_like(u)
    for k in range(1, len(u)):
        y[k] = 0.8 * y[k - 1] + 0.2 * u[k]
    model = fit_discrete_transfer(u, y, fs=100, na=1, nb=0)
    assert model.transfer_function.denominator.shape == (2,)
    assert model.transfer_function.numerator.shape == (1,)
    assert model.transfer_function.domain_variable == "z^-1"


def test_tfestimate_returns_matching_arrays():
    u = np.sin(np.linspace(0, 20, 512))
    f, h, c = tfestimate(u, 2 * u, fs=100)
    assert f.shape == h.shape == c.shape
