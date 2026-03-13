"""
Shared fixtures for ZIT-DGLM tests.

All fixtures generate synthetic data from a known DGP, allowing verification
of parameter recovery and correct model behaviour.

DGP used throughout:
    Features: x1 ~ N(0,1), x2 ~ N(0,1), x3 ~ Bernoulli(0.5)
    True functions:
        log(mu_true) = 0.5 + 0.8*x1 - 0.4*x2
        log(phi_true) = 0.2 + 0.3*|x2|
        logit(q_true) = -1.0 + 0.6*x1
    Structural zero: z_i ~ Bernoulli(q_true_i)
    y_i = 0                      if z_i = 1
    y_i ~ Tweedie(mu_true_i, phi_true_i, p=1.5) if z_i = 0
"""

import numpy as np
import polars as pl
import pytest
from typing import Tuple


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def simulate_zit_data(
    n: int = 500,
    tweedie_power: float = 1.5,
    seed: int = 42,
    include_exposure: bool = False,
    zero_rate: float | None = None,
    all_zeros: bool = False,
    no_zeros: bool = False,
) -> Tuple[pl.DataFrame, pl.Series, dict]:
    """
    Generate synthetic ZIT data from a known DGP.

    Returns
    -------
    X:
        Feature DataFrame.
    y:
        Observed aggregate losses.
    truth:
        Dict with true parameter arrays (mu, phi, q) for verification.
    """
    rng = np.random.default_rng(seed)

    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    x3 = rng.integers(0, 2, n).astype(float)

    if include_exposure:
        exposure = rng.uniform(0.5, 2.0, n)
    else:
        exposure = np.ones(n)

    # True parameter functions
    log_mu = 0.5 + 0.8 * x1 - 0.4 * x2
    mu_true = np.exp(log_mu)

    log_phi = 0.2 + 0.3 * np.abs(x2)
    phi_true = np.exp(log_phi)

    if zero_rate is not None:
        # Override q with fixed zero rate for edge case testing
        q_true = np.full(n, zero_rate)
    else:
        logit_q = -1.0 + 0.6 * x1
        q_true = sigmoid(logit_q)

    if all_zeros:
        q_true = np.ones(n)
    if no_zeros:
        q_true = np.zeros(n)

    # Draw structural zeros
    z = rng.binomial(1, q_true)

    # Draw Tweedie outcomes for non-structural zeros
    # Tweedie(mu, phi, p) is compound Poisson-Gamma:
    #   N ~ Poisson(lambda) where lambda = w * mu^(2-p) / (phi*(2-p))
    #   Given N: Y = sum of N Gamma random variables, each Gamma(shape=2-p/(p-1), scale=phi*(p-1)*mu^(p-1))
    p = tweedie_power
    p2 = 2.0 - p
    p1 = p - 1.0  # p - 1

    lam = exposure * (mu_true ** p2) / (phi_true * p2)
    # Gamma shape/scale parameterisation: E[X] = shape*scale = mu_gamma, Var = shape*scale^2
    # Each individual claim: shape = (2-p)/(p-1), scale = phi*(p-1)*mu^(p-1)/w (per-unit)
    gamma_shape = p2 / p1
    gamma_scale = phi_true * p1 * (mu_true ** p1) / exposure

    y = np.zeros(n)
    for i in range(n):
        if z[i] == 1:
            y[i] = 0.0
        else:
            n_claims = rng.poisson(lam[i])
            if n_claims == 0:
                y[i] = 0.0
            else:
                # Sum of n_claims independent Gamma(gamma_shape, gamma_scale) r.v.s
                claims = rng.gamma(gamma_shape, gamma_scale[i], n_claims)
                y[i] = float(np.sum(claims))

    feature_dict = {"x1": x1, "x2": x2, "x3": x3}
    if include_exposure:
        feature_dict["exposure"] = exposure

    X = pl.DataFrame(feature_dict)
    y_series = pl.Series(y)

    truth = {
        "mu": mu_true,
        "phi": phi_true,
        "q": q_true,
        "exposure": exposure,
        "z": z,
    }

    return X, y_series, truth


@pytest.fixture
def small_zit_data():
    """Small dataset (n=200) for fast tests."""
    return simulate_zit_data(n=200, seed=0)


@pytest.fixture
def medium_zit_data():
    """Medium dataset (n=500) for accuracy tests."""
    return simulate_zit_data(n=500, seed=1)


@pytest.fixture
def large_zit_data():
    """Larger dataset (n=1000) for EM convergence tests."""
    return simulate_zit_data(n=1000, seed=2)


@pytest.fixture
def zit_data_with_exposure():
    """Dataset with variable exposure weights."""
    return simulate_zit_data(n=400, seed=3, include_exposure=True)


@pytest.fixture
def all_zeros_data():
    """Edge case: all observations are zero (q=1)."""
    return simulate_zit_data(n=100, seed=4, all_zeros=True)


@pytest.fixture
def no_zeros_data():
    """Edge case: no zero observations (q=0, standard Tweedie)."""
    return simulate_zit_data(n=200, seed=5, no_zeros=True)


@pytest.fixture
def low_zero_rate_data():
    """Dataset with low structural zero rate (~5%)."""
    return simulate_zit_data(n=300, seed=6, zero_rate=0.05)


@pytest.fixture
def high_zero_rate_data():
    """Dataset with high structural zero rate (~70%)."""
    return simulate_zit_data(n=300, seed=7, zero_rate=0.70)


@pytest.fixture
def single_feature_data():
    """Edge case: single numeric feature."""
    rng = np.random.default_rng(8)
    n = 200
    x = rng.normal(0, 1, n)
    mu = np.exp(0.3 * x + 0.5)
    q = sigmoid(-0.5 + 0.4 * x)
    z = rng.binomial(1, q)
    y = np.where(z == 1, 0.0, rng.exponential(mu))

    X = pl.DataFrame({"x1": x})
    y_series = pl.Series(y)
    truth = {"mu": mu, "q": q}
    return X, y_series, truth


@pytest.fixture
def fitted_small_model(small_zit_data):
    """Pre-fitted ZITModel on small data (n_estimators=5 for speed)."""
    from insurance_zit_dglm.model import ZITModel

    X, y, _ = small_zit_data
    model = ZITModel(n_estimators=5, em_iterations=3, verbose=0)
    model.fit(X, y)
    return model, X, y
