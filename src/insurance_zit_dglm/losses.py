"""
CatBoost custom objectives for the three ZIT-DGLM heads.

Each objective implements calc_ders_range(approxes, targets, weights) returning
a list of (neg_gradient, neg_hessian) tuples — the sign convention required by CatBoost.

Sources:
    So & Valdez arXiv:2406.16206 / NAAJ 2025 — gradient/Hessian derivations for ZIT trees
    Gu arXiv:2405.14990 — EM weighting scheme for dispersion head
"""

from __future__ import annotations

import numpy as np
from typing import Sequence


# ---------------------------------------------------------------------------
# Head 1: Mean model (mu = exp(F_mu), log link)
# ---------------------------------------------------------------------------


class ZITTweedieLoss:
    """
    CatBoost custom loss for the mean head of ZIT-DGLM.

    Implements the ZIT Tweedie negative log-likelihood gradient and Hessian
    with respect to the raw (log-scale) boosting score F_mu, where mu = exp(F_mu).

    Parameters
    ----------
    tweedie_power:
        Tweedie power parameter p in (1, 2). Compound Poisson-Gamma regime.
    phi_values:
        Per-observation dispersion values phi_i (updated each EM iteration).
    q_values:
        Per-observation structural zero probabilities q_i (updated each EM iteration).
    em_weights:
        Per-observation EM posterior weights (1 - Pi_i^k). Observations likely to be
        structural zeros receive low weight on the Tweedie likelihood.
    exposure:
        Per-observation exposure weights w_i.
    """

    def __init__(
        self,
        tweedie_power: float,
        phi_values: np.ndarray,
        q_values: np.ndarray,
        em_weights: np.ndarray,
        exposure: np.ndarray,
    ) -> None:
        self.p = tweedie_power
        self.phi = phi_values
        self.q = q_values
        self.em_w = em_weights
        self.exposure = exposure

    def calc_ders_range(
        self,
        approxes: list[float],
        targets: list[float],
        weights: list[float] | None,
    ) -> list[tuple[float, float]]:
        """Return per-observation (neg_gradient, neg_hessian) for CatBoost."""
        approxes_arr = np.array(approxes)
        targets_arr = np.array(targets)

        mu = np.exp(approxes_arr)
        y = targets_arr
        p = self.p
        phi = self.phi
        q = self.q
        em_w = self.em_w
        w = self.exposure

        result = []
        for i in range(len(approxes_arr)):
            g, h = _zit_tweedie_ders(
                mu=mu[i],
                y=y[i],
                p=p,
                phi=phi[i],
                q=q[i],
                em_weight=em_w[i],
                exposure=w[i],
            )
            result.append((g, h))

        return result

    def is_max_optimal(self) -> bool:
        return False


def _zit_tweedie_ders(
    mu: float,
    y: float,
    p: float,
    phi: float,
    q: float,
    em_weight: float,
    exposure: float,
) -> tuple[float, float]:
    """
    Compute (neg_gradient, neg_hessian) for one observation under the ZIT Tweedie loss.

    Formulae from So & Valdez arXiv:2406.16206, adapted for exposure weights.

    For y = 0:
        alpha = (1 - q) * exp(-w * mu^(2-p) / (phi * (2-p)))
        beta  = w * mu^(2-p) / phi
        g = alpha * beta / (q + alpha)
        h = alpha * beta * [(2-p-beta)*(q+alpha) + alpha*beta] / (q+alpha)^2

    For y > 0:
        beta = w * mu^(2-p) / phi
        g = -(w/phi) * y * mu^(1-p) + beta
        h = -(w/phi) * y * (1-p) * mu^(1-p) + (2-p) * beta

    All gradients are wrt F_mu (log scale); CatBoost multiplies by the Jacobian
    (mu) internally when using raw approxes.

    Returns the NEGATIVE gradient and NEGATIVE Hessian (CatBoost convention).
    """
    eps = 1e-10
    mu = max(mu, eps)
    phi = max(phi, eps)
    p2 = 2.0 - p  # (2-p) shorthand

    beta = exposure * (mu ** p2) / phi

    if y <= 0.0:
        # Zero observation: mixture of structural zero and Tweedie zero
        tweedie_zero_log = -exposure * (mu ** p2) / (phi * p2)
        alpha = (1.0 - q) * np.exp(tweedie_zero_log)
        denom = q + alpha
        denom = max(denom, eps)

        g = alpha * beta / denom
        h = alpha * beta * ((p2 - beta) * denom + alpha * beta) / (denom ** 2)
    else:
        # Positive observation: purely from Tweedie component
        g = -(exposure / phi) * y * (mu ** (1.0 - p)) + beta
        h = -(exposure / phi) * y * (1.0 - p) * (mu ** (1.0 - p)) + p2 * beta

    # Apply EM weight: down-weight observations likely to be structural zeros
    g *= em_weight
    h *= em_weight

    # CatBoost expects negative gradient and negative Hessian
    # Hessian must be positive for stable training
    h = max(h, eps)

    return (-g, -h)


# ---------------------------------------------------------------------------
# Head 2: Zero-inflation model (q = sigmoid(F_pi), logit link)
# ---------------------------------------------------------------------------


class ZITZeroInflationLoss:
    """
    CatBoost custom loss for the zero-inflation head of ZIT-DGLM.

    Implements weighted binary cross-entropy with EM posterior soft labels Pi_i.
    For y_i > 0: Pi_i = 0 (observation is certainly not a structural zero).
    For y_i = 0: Pi_i in [0,1] from the E-step.

    The gradient is q_i - Pi_i (binary cross-entropy with soft label Pi_i).
    """

    def __init__(self, pi_em_weights: np.ndarray) -> None:
        """
        Parameters
        ----------
        pi_em_weights:
            Per-observation EM posterior weights Pi_i^k. Pi_i = 0 for y_i > 0.
        """
        self.pi_em = pi_em_weights

    def calc_ders_range(
        self,
        approxes: list[float],
        targets: list[float],
        weights: list[float] | None,
    ) -> list[tuple[float, float]]:
        """Return per-observation (neg_gradient, neg_hessian)."""
        approxes_arr = np.array(approxes)
        pi_em = self.pi_em
        eps = 1e-10

        # sigmoid of raw score
        q = 1.0 / (1.0 + np.exp(-approxes_arr))
        q = np.clip(q, eps, 1.0 - eps)

        # gradient of binary cross-entropy with soft label: d/dF = q - Pi
        grad = q - pi_em  # shape (n,)

        # Hessian: q * (1 - q)
        hess = q * (1.0 - q)
        hess = np.maximum(hess, eps)

        # CatBoost convention: return negative grad, negative hess
        return [(-grad[i], -hess[i]) for i in range(len(grad))]

    def is_max_optimal(self) -> bool:
        return False


# ---------------------------------------------------------------------------
# Head 3: Dispersion model (phi = exp(F_phi), log link)
# ---------------------------------------------------------------------------


class ZITDispersionLoss:
    """
    CatBoost custom loss for the dispersion head of ZIT-DGLM.

    Implements a pseudo-gamma log-likelihood following the Smyth-Jorgensen
    extended quasi-likelihood trick. The dispersion phi_i is modelled as if the
    scaled unit deviance d_i = D(y_i; mu_hat_i) / w_i follows a Gamma(1/2, phi_i)
    distribution.

    Loss per observation:
        l_i = (1 - Pi_i) * [- d_i / (2 * phi_i) - (1/2) * log(phi_i)]

    With phi_i = exp(F_phi_i), log link:
        d/dF_phi l_i = (1 - Pi_i) * [d_i / (2 * phi_i) - 1/2]
        d2/dF_phi2 l_i = (1 - Pi_i) * [- d_i / (2 * phi_i) + 1/2]

    CatBoost sees this as a regression of pseudo-responses with EM weights.
    """

    def __init__(
        self,
        unit_deviances: np.ndarray,
        em_weights: np.ndarray,
    ) -> None:
        """
        Parameters
        ----------
        unit_deviances:
            Scaled unit deviances d_i = D_zeta(y_i; mu_hat_i) / w_i.
        em_weights:
            EM weights (1 - Pi_i^k). Downweights likely structural zeros.
        """
        self.d = unit_deviances
        self.em_w = em_weights

    def calc_ders_range(
        self,
        approxes: list[float],
        targets: list[float],
        weights: list[float] | None,
    ) -> list[tuple[float, float]]:
        """Return per-observation (neg_gradient, neg_hessian)."""
        approxes_arr = np.array(approxes)
        eps = 1e-10

        phi = np.exp(approxes_arr)
        phi = np.maximum(phi, eps)

        d = self.d
        em_w = self.em_w

        # Gradient of log-likelihood wrt F_phi (log link)
        grad = em_w * (d / (2.0 * phi) - 0.5)

        # Hessian (second derivative wrt F_phi)
        # d2/dF2 [em_w * (-d/(2*phi) - 0.5*log(phi))]
        # phi = exp(F), d(phi)/dF = phi
        # d2l/dF2 = em_w * (-d/(2*phi) + 0) ... wait, full chain:
        # l = em_w * [-d/(2*phi) - 0.5*F]
        # dl/dF = em_w * [d/(2*phi) - 0.5]  (since d(1/phi)/dF = -1/phi)
        # d2l/dF2 = em_w * [-d/(2*phi)]
        hess = em_w * (-d / (2.0 * phi))
        # Hessian must be positive for stable boosting (we negate for CatBoost)
        hess_neg = np.maximum(-hess, eps)

        return [(-grad[i], hess_neg[i]) for i in range(len(grad))]

    def is_max_optimal(self) -> bool:
        return False


# ---------------------------------------------------------------------------
# Utility: unit deviance computation
# ---------------------------------------------------------------------------


def tweedie_unit_deviance(
    y: np.ndarray,
    mu: np.ndarray,
    p: float,
    exposure: np.ndarray | None = None,
) -> np.ndarray:
    """
    Compute the Tweedie unit deviance D_p(y; mu) per observation.

    D_p(y; mu) = 2 * [y * (y^(1-p) - mu^(1-p)) / (1-p)
                      - (y^(2-p) - mu^(2-p)) / (2-p)]

    For y = 0, the first term is 0 by convention (taking the limit y -> 0+).

    Parameters
    ----------
    y:
        Observed values (>= 0).
    mu:
        Predicted mean values (> 0).
    p:
        Tweedie power parameter in (1, 2).
    exposure:
        Exposure weights w_i. If provided, returns scaled deviance d_i = D_i / w_i.

    Returns
    -------
    numpy.ndarray
        Per-observation unit deviances (or scaled deviances if exposure given).
    """
    eps = 1e-10
    mu = np.maximum(mu, eps)
    p1 = 1.0 - p  # (1-p)
    p2 = 2.0 - p  # (2-p)

    y_pos = np.where(y > 0, y, 0.0)

    # First term: y * (y^(1-p) - mu^(1-p)) / (1-p) — zero when y=0
    t1 = np.where(
        y > 0,
        y_pos * ((y_pos ** p1) - (mu ** p1)) / p1,
        0.0,
    )

    # Second term: (y^(2-p) - mu^(2-p)) / (2-p)
    t2 = np.where(
        y > 0,
        ((y_pos ** p2) - (mu ** p2)) / p2,
        -(mu ** p2) / p2,
    )

    deviance = 2.0 * (t1 - t2)
    deviance = np.maximum(deviance, 0.0)

    if exposure is not None:
        exposure = np.maximum(exposure, eps)
        deviance = deviance / exposure

    return deviance


def zit_log_likelihood(
    y: np.ndarray,
    mu: np.ndarray,
    phi: np.ndarray,
    q: np.ndarray,
    p: float,
    exposure: np.ndarray | None = None,
) -> np.ndarray:
    """
    Compute the ZIT log-likelihood contribution per observation.

    For y_i = 0:
        log(q_i + (1-q_i)*exp(-w_i*mu_i^(2-p)/(phi_i*(2-p))))
    For y_i > 0:
        log(1-q_i) + (w_i/phi_i)*[y_i*mu_i^(1-p)/(1-p) - mu_i^(2-p)/(2-p)]
        (normalising constant omitted as it does not depend on parameters)

    Parameters
    ----------
    y:
        Observed values.
    mu:
        Predicted Tweedie mean (conditional on non-structural-zero).
    phi:
        Predicted dispersion.
    q:
        Predicted structural zero probability.
    p:
        Tweedie power.
    exposure:
        Per-observation exposure weights w_i.

    Returns
    -------
    numpy.ndarray
        Per-observation log-likelihood values (normalising constant excluded).
    """
    eps = 1e-10
    n = len(y)
    if exposure is None:
        exposure = np.ones(n)

    mu = np.maximum(mu, eps)
    phi = np.maximum(phi, eps)
    q = np.clip(q, eps, 1.0 - eps)

    p2 = 2.0 - p
    p1 = 1.0 - p

    ll = np.empty(n)
    for i in range(n):
        w = exposure[i]
        if y[i] <= 0.0:
            tweedie_zero_log = -w * (mu[i] ** p2) / (phi[i] * p2)
            ll[i] = np.log(q[i] + (1.0 - q[i]) * np.exp(tweedie_zero_log))
        else:
            tweedie_pos = (w / phi[i]) * (
                y[i] * (mu[i] ** p1) / p1 - (mu[i] ** p2) / p2
            )
            ll[i] = np.log(1.0 - q[i]) + tweedie_pos

    return ll
