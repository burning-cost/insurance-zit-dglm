"""
Tweedie power parameter estimation by profile likelihood.

The power parameter p in (1, 2) is not gradient-boosted — it is estimated
separately by searching over a grid of candidate values and fitting a simple
ZIT model at each. The value maximising the observed log-likelihood is returned.

This follows the convention of Jorgensen (1997) and So & Valdez (NAAJ 2025)
for the compound Poisson-Gamma regime (p in (1, 2)).

Reference: Dunn & Smyth (2005), Saddlepoint approximations for Tweedie families.
"""

from __future__ import annotations

import numpy as np
from typing import Sequence


def estimate_power(
    y: np.ndarray,
    mu: np.ndarray,
    phi: np.ndarray | float = 1.0,
    q: np.ndarray | None = None,
    p_grid: Sequence[float] | None = None,
    exposure: np.ndarray | None = None,
) -> float:
    """
    Estimate the Tweedie power parameter by profile likelihood.

    Searches over p_grid and returns the value with highest ZIT log-likelihood,
    using fixed mu, phi, and q estimates. Typically called after a first-pass ZIT
    fit with p=1.5 to refine the power estimate.

    Parameters
    ----------
    y:
        Observed aggregate losses (>= 0).
    mu:
        Fitted mean predictions (Tweedie component mean, conditional on non-structural-zero).
    phi:
        Fitted dispersion (scalar or per-observation array).
    q:
        Fitted structural zero probabilities. If None, assumes q = 0 (standard Tweedie).
    p_grid:
        Grid of power values to search. Defaults to [1.1, 1.2, ..., 1.9].
    exposure:
        Per-observation exposure weights w_i. Defaults to ones.

    Returns
    -------
    float
        Power value from p_grid maximising the profile log-likelihood.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(42)
    >>> y = np.concatenate([np.zeros(300), rng.exponential(0.5, 700)])
    >>> mu = np.ones(1000) * 0.7
    >>> p_hat = estimate_power(y, mu)
    >>> 1.0 < p_hat < 2.0
    True
    """
    n = len(y)
    if exposure is None:
        exposure = np.ones(n)
    if q is None:
        q = np.zeros(n)
    if p_grid is None:
        p_grid = [round(1.0 + 0.1 * k, 1) for k in range(1, 10)]  # 1.1 to 1.9

    if np.isscalar(phi):
        phi_arr = np.full(n, float(phi))
    else:
        phi_arr = np.asarray(phi, dtype=float)

    q_arr = np.asarray(q, dtype=float)
    mu_arr = np.asarray(mu, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    exposure_arr = np.asarray(exposure, dtype=float)

    best_p = float(p_grid[0])
    best_ll = -np.inf

    for p in p_grid:
        ll = _profile_ll(y_arr, mu_arr, phi_arr, q_arr, float(p), exposure_arr)
        if ll > best_ll:
            best_ll = ll
            best_p = float(p)

    return best_p


def _profile_ll(
    y: np.ndarray,
    mu: np.ndarray,
    phi: np.ndarray,
    q: np.ndarray,
    p: float,
    exposure: np.ndarray,
) -> float:
    """
    Compute the total ZIT log-likelihood for a given power parameter p.

    Normalising constants (which depend on p through infinite series) are omitted
    as they shift the log-likelihood uniformly across observations.
    """
    eps = 1e-10
    p2 = 2.0 - p
    p1 = 1.0 - p

    mu_safe = np.maximum(mu, eps)
    phi_safe = np.maximum(phi, eps)
    q_safe = np.clip(q, eps, 1.0 - eps)

    ll = 0.0
    for i in range(len(y)):
        w = exposure[i]
        if y[i] <= 0.0:
            log_tweedie_zero = -w * (mu_safe[i] ** p2) / (phi_safe[i] * p2)
            log_tweedie_zero = max(log_tweedie_zero, -700.0)
            prob_zero = q_safe[i] + (1.0 - q_safe[i]) * np.exp(log_tweedie_zero)
            ll += np.log(max(prob_zero, eps))
        else:
            tweedie_ll = (w / phi_safe[i]) * (
                y[i] * (mu_safe[i] ** p1) / p1 - (mu_safe[i] ** p2) / p2
            )
            ll += np.log(1.0 - q_safe[i]) + tweedie_ll

    return ll


def power_grid_search(
    y: np.ndarray,
    fit_fn: "Callable",
    p_grid: Sequence[float] | None = None,
    exposure: np.ndarray | None = None,
) -> dict:
    """
    Full profile likelihood search, fitting a model at each candidate power value.

    Unlike estimate_power() which uses fixed mu/phi/q, this refits the model
    entirely at each p value. Computationally expensive but more accurate.

    Parameters
    ----------
    y:
        Observed aggregate losses.
    fit_fn:
        Callable fit_fn(p) -> (mu, phi, q) returning predictions for each power.
    p_grid:
        Grid of power values to search.
    exposure:
        Per-observation exposure weights.

    Returns
    -------
    dict
        Keys: 'best_p', 'p_grid', 'log_likelihoods', 'best_ll'.
    """
    n = len(y)
    if exposure is None:
        exposure = np.ones(n)
    if p_grid is None:
        p_grid = [round(1.0 + 0.1 * k, 1) for k in range(1, 10)]

    log_likelihoods = []
    for p in p_grid:
        mu, phi, q = fit_fn(p)
        ll = _profile_ll(
            np.asarray(y),
            np.asarray(mu),
            np.asarray(phi),
            np.asarray(q),
            float(p),
            np.asarray(exposure),
        )
        log_likelihoods.append(ll)

    best_idx = int(np.argmax(log_likelihoods))
    return {
        "best_p": float(p_grid[best_idx]),
        "p_grid": list(p_grid),
        "log_likelihoods": log_likelihoods,
        "best_ll": log_likelihoods[best_idx],
    }
