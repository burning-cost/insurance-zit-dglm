"""
EM algorithm orchestration for ZIT-DGLM.

Implements Algorithm 1 from Gu (arXiv:2405.14990):
    Repeat until convergence:
        E-step: compute posterior structural-zero probability Pi_i for each observation
        M-step 1: update pi (zero-inflation) model with EM-weighted logistic loss
        M-step 2: update mu (mean) model with EM-weighted ZIT Tweedie loss
        M-step 3: update phi (dispersion) model with gamma pseudo-likelihood

The three sub-steps within each M-step run sequentially; the outer loop runs
until the observed log-likelihood changes by less than em_tol.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class EMState:
    """Carries the current parameter estimates across EM iterations."""

    mu: np.ndarray          # Current mu predictions, shape (n,)
    phi: np.ndarray         # Current phi predictions, shape (n,)
    q: np.ndarray           # Current structural zero probabilities, shape (n,)
    pi_em: np.ndarray       # E-step posterior Pi_i, shape (n,)
    em_weights: np.ndarray  # (1 - Pi_i), weight for Tweedie component, shape (n,)
    log_likelihoods: list[float] = field(default_factory=list)


def e_step(
    y: np.ndarray,
    mu: np.ndarray,
    phi: np.ndarray,
    q: np.ndarray,
    p: float,
    exposure: np.ndarray,
) -> np.ndarray:
    """
    E-step: compute posterior probability Pi_i that observation i is a structural zero.

    For y_i > 0: Pi_i = 0 (certainty — structural zeros cannot have positive values).
    For y_i = 0: Pi_i = pi_i / (pi_i + (1-pi_i)*exp(-w_i*mu_i^(2-p)/(phi_i*(2-p))))

    The exposure weight w_i enters inside the Tweedie zero probability:
        P(Tweedie=0 | mu, phi, w) = exp(-w * mu^(2-p) / (phi*(2-p)))
    Higher exposure means lower probability of a Tweedie zero, pushing Pi upward.

    Parameters
    ----------
    y:
        Observed values.
    mu:
        Current mean predictions (Tweedie component).
    phi:
        Current dispersion predictions.
    q:
        Current structural zero probabilities.
    p:
        Tweedie power parameter.
    exposure:
        Per-observation exposure weights w_i.

    Returns
    -------
    numpy.ndarray
        Posterior structural zero probabilities Pi_i in [0, 1].
    """
    eps = 1e-10
    n = len(y)
    pi = np.zeros(n)

    mu_safe = np.maximum(mu, eps)
    phi_safe = np.maximum(phi, eps)
    q_safe = np.clip(q, eps, 1.0 - eps)

    p2 = 2.0 - p

    zero_mask = y <= 0.0
    if not np.any(zero_mask):
        return pi

    mu_z = mu_safe[zero_mask]
    phi_z = phi_safe[zero_mask]
    q_z = q_safe[zero_mask]
    w_z = exposure[zero_mask]

    # Tweedie zero probability for zero observations
    log_tweedie_zero = -w_z * (mu_z ** p2) / (phi_z * p2)
    # Clip to avoid underflow
    log_tweedie_zero = np.maximum(log_tweedie_zero, -700.0)
    tweedie_zero_prob = np.exp(log_tweedie_zero)

    # P(structural zero) / P(Y=0)
    numerator = q_z
    denominator = q_z + (1.0 - q_z) * tweedie_zero_prob
    denominator = np.maximum(denominator, eps)

    pi[zero_mask] = numerator / denominator
    pi = np.clip(pi, 0.0, 1.0)

    return pi


def initialise_state(
    y: np.ndarray,
    exposure: np.ndarray,
    tweedie_power: float,
) -> dict:
    """
    Compute initial parameter guesses for the EM algorithm.

    Initialisation strategy (Gu 2405.14990, practical implementation):
        - mu_0: mean of positive observations (or 1.0 if no positives)
        - phi_0: 1.0 constant (log(phi)=0)
        - q_0: estimated from excess zero rate beyond what compound Poisson predicts
               If observed_zero_rate <= CP_zero_rate: q_0 = eps (no structural zeros)
               Otherwise: q_0 = (observed_zero_rate - CP_zero_rate) / (1 - CP_zero_rate)
    """
    eps = 1e-6
    p2 = 2.0 - tweedie_power

    zero_mask = y <= 0.0
    obs_zero_rate = float(np.mean(zero_mask))

    pos_y = y[y > 0.0]
    mu_0 = float(np.mean(pos_y)) if len(pos_y) > 0 else 1.0

    # Compound Poisson zero probability at initial mu=mu_0, phi=1
    cp_zero = np.exp(-(mu_0 ** p2) / p2)
    q_0 = max((obs_zero_rate - cp_zero) / max(1.0 - cp_zero, eps), eps)
    q_0 = min(q_0, 0.9)  # cap to avoid degenerate initialisation

    return {
        "mu_0": mu_0,
        "phi_0": 1.0,
        "q_0": q_0,
    }


def check_convergence(
    log_likelihoods: list[float],
    tol: float,
) -> bool:
    """
    Check convergence based on relative change in observed log-likelihood.

    Convergence when |ell^(k) - ell^(k-1)| / (|ell^(k-1)| + 1) < tol.
    Relative criterion avoids scale dependence across datasets.
    """
    if len(log_likelihoods) < 2:
        return False
    ell_prev = log_likelihoods[-2]
    ell_curr = log_likelihoods[-1]
    rel_change = abs(ell_curr - ell_prev) / (abs(ell_prev) + 1.0)
    return rel_change < tol
