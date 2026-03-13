"""
CatBoost custom objectives for the three ZIT-DGLM heads.

Each objective implements calc_ders_range(approxes, targets, weights) returning
a list of (der1, der2) tuples — where der1 is the gradient of the LOSS function
(what CatBoost minimises) and der2 is the second derivative (always positive).

Since CatBoost minimises loss = -LL:
    der1 = d(Loss)/dF = -dLL/dF
    der2 = d2(Loss)/dF2 = -d2LL/dF2  (positive, since LL is concave)

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
        """Return per-observation (der1, der2) for CatBoost.

        der1 = d(NegLL)/dF = gradient of loss function
        der2 = d2(NegLL)/dF2 = second derivative of loss (positive)
        """
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
    Compute (der1, der2) for one observation under the ZIT Tweedie loss.

    der1 = d(NegLL)/dF, der2 = d2(NegLL)/dF2.

    Formulae from So & Valdez arXiv:2406.16206, adapted for exposure weights.

    For y = 0:
        alpha = (1 - q) * exp(-w * mu^(2-p) / (phi * (2-p)))
        beta  = w * mu^(2-p) / phi
        der1 = alpha * beta / (q + alpha)
        der2 = alpha * beta * [(2-p-beta)*(q+alpha) + alpha*beta] / (q+alpha)^2

    For y > 0:
        beta = w * mu^(2-p) / phi
        der1 = -(w/phi) * y * mu^(1-p) + beta    [note: this equals -dLL/dF]

        Wait, for y > 0:
        LL = (w/phi) * [y * mu^(1-p)/(1-p) - mu^(2-p)/(2-p)]
        dLL/dF = (w/phi) * [y*(1-p)*mu^(1-p)/(1-p) - (2-p)*mu^(2-p)/(2-p)]
               = (w/phi) * [y*mu^(1-p) - mu^(2-p)]  (Jacobian: dmu/dF = mu)
               Wait: dLL/dmu = (w/phi)*[y*mu^(-p) - mu^(1-p)]
               dLL/dF = (w/phi)*mu*[y*mu^(-p) - mu^(1-p)]
                      = (w/phi)*[y*mu^(1-p) - mu^(2-p)]
               NegLL der1 = -dLL/dF = (w/phi)*[mu^(2-p) - y*mu^(1-p)]
                          = beta - (w/phi)*y*mu^(1-p)

    All gradients are wrt F_mu (log scale).
    Returns (der1, der2) where der1 = d(NegLL)/dF, der2 = d2(NegLL)/dF2 > 0.
    """
    eps = 1e-10
    mu = max(mu, eps)
    phi = max(phi, eps)
    p2 = 2.0 - p  # (2-p) shorthand
    p1 = 1.0 - p  # (1-p) shorthand

    beta = exposure * (mu ** p2) / phi

    if y <= 0.0:
        # Zero observation: mixture of structural zero and Tweedie zero
        tweedie_zero_log = -exposure * (mu ** p2) / (phi * p2)
        alpha = (1.0 - q) * np.exp(tweedie_zero_log)
        denom = q + alpha
        denom = max(denom, eps)

        # der1 = d(NegLL)/dF for y=0
        # NegLL = -log(q + alpha), dNegLL/dF = -d(log(q+alpha))/dF
        # d(alpha)/dF = alpha * (-beta) (since d(exp(-beta))/dF = exp(-beta)*(-dbeta/dF))
        #             and dbeta/dF = p2*beta (since beta = w*mu^p2/phi and dmu/dF = mu)
        # d(log(q+alpha))/dF = (d(alpha)/dF) / (q+alpha) = alpha * (-p2*beta) / denom
        # But der1 = alpha*beta/denom is derived from the derivative chain:
        # dLL/dF = alpha*(-dbeta/dF*p2^{-1}*...) ... let me use the clean formula:
        # NegLL = -log(q + (1-q)*exp(-w*mu^p2/(phi*p2)))
        # Let t = w*mu^p2/(phi*p2), so alpha = (1-q)*exp(-t)
        # dNegLL/dF = (1-q)*exp(-t) * (dt/dF) / (q + (1-q)*exp(-t))
        # dt/dF = d(w*mu^p2/(phi*p2))/dF = w*p2*mu^p2/(phi*p2) = w*mu^p2/phi = beta
        # So dNegLL/dF = alpha * beta / denom
        der1 = alpha * beta / denom

        # der2 = d2(NegLL)/dF2
        # d(alpha*beta/denom)/dF:
        # Let num = alpha*beta, denom = q+alpha
        # d(num)/dF = alpha*(dbeta/dF) + beta*(d(alpha)/dF)
        #           = alpha*p2*beta + beta*(-alpha*beta)  [chain rule, but need to be careful]
        # Actually d(alpha)/dF = alpha * d(-t)/dF = -alpha*beta
        # d(beta)/dF = p2*beta
        # d(alpha*beta)/dF = beta*d(alpha)/dF + alpha*d(beta)/dF
        #                  = -alpha*beta^2 + alpha*p2*beta = alpha*beta*(p2-beta)
        # d(denom)/dF = d(alpha)/dF = -alpha*beta
        # d(alpha*beta/denom)/dF = [d(num)/dF*denom - num*d(denom)/dF] / denom^2
        #   = [alpha*beta*(p2-beta)*denom - alpha*beta*(-alpha*beta)] / denom^2
        #   = alpha*beta * [(p2-beta)*denom + alpha*beta] / denom^2
        der2 = alpha * beta * ((p2 - beta) * denom + alpha * beta) / (denom ** 2)
    else:
        # Positive observation: purely from Tweedie component
        # LL = (w/phi)*[y*mu^(1-p)/(1-p) - mu^(2-p)/(2-p)] + log(1-q)
        # dLL/dF = (w/phi)*[y*mu^(1-p) - mu^(2-p)] * (from dmu/dF=mu)
        #        = (w/phi) * y * mu^(1-p) - beta
        # der1 = -dLL/dF = -(w/phi)*y*mu^(1-p) + beta = beta - (w/phi)*y*mu^(1-p)
        der1 = beta - (exposure / phi) * y * (mu ** p1)

        # d2LL/dF2 = (w/phi)*[y*(1-p)*mu^(1-p) - p2*mu^(2-p)]
        # der2 = -d2LL/dF2 = -(w/phi)*y*(1-p)*mu^(1-p) + p2*beta
        der2 = p2 * beta - (exposure / phi) * y * p1 * (mu ** p1)

    # Apply EM weight: down-weight observations likely to be structural zeros
    der1 *= em_weight
    der2 *= em_weight

    # Second derivative must be positive for stable training
    der2 = max(der2, eps)

    return (der1, der2)


# ---------------------------------------------------------------------------
# Head 2: Zero-inflation model (q = sigmoid(F_pi), logit link)
# ---------------------------------------------------------------------------


class ZITZeroInflationLoss:
    """
    CatBoost custom loss for the zero-inflation head of ZIT-DGLM.

    Implements weighted binary cross-entropy with EM posterior soft labels Pi_i.
    For y_i > 0: Pi_i = 0 (observation is certainly not a structural zero).
    For y_i = 0: Pi_i in [0,1] from the E-step.

    Loss = -[Pi*log(q) + (1-Pi)*log(1-q)]  (minimised)
    der1 = d(Loss)/dF = q - Pi
    der2 = d2(Loss)/dF2 = q*(1-q)
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
        """Return per-observation (der1, der2).

        der1 = d(Loss)/dF = q - Pi (gradient of NegCrossEntropy)
        der2 = q*(1-q) (positive, second derivative of NegCE)
        """
        approxes_arr = np.array(approxes)
        pi_em = self.pi_em
        eps = 1e-10

        # sigmoid of raw score
        q = 1.0 / (1.0 + np.exp(-approxes_arr))
        q = np.clip(q, eps, 1.0 - eps)

        # der1 = d(NegCE)/dF = q - Pi
        der1 = q - pi_em

        # der2 = q * (1 - q)
        der2 = np.maximum(q * (1.0 - q), eps)

        return [(der1[i], der2[i]) for i in range(len(der1))]

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

    Loss (NegLL) per observation:
        L_i = (1 - Pi_i) * [d_i / (2 * phi_i) + (1/2) * log(phi_i)]

    With phi_i = exp(F_phi_i), log link:
        der1 = dL/dF = (1 - Pi_i) * [1/2 - d_i / (2 * phi_i)]
        der2 = d2L/dF2 = (1 - Pi_i) * d_i / (2 * phi_i)  (positive when d>0)
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
        """Return per-observation (der1, der2).

        der1 = d(Loss)/dF = em_w * (0.5 - d/(2*phi))
        der2 = d2(Loss)/dF2 = em_w * d/(2*phi) >= 0
        """
        approxes_arr = np.array(approxes)
        eps = 1e-10

        phi = np.exp(approxes_arr)
        phi = np.maximum(phi, eps)

        d = self.d
        em_w = self.em_w

        # NegLL = em_w * [d/(2*phi) + 0.5*log(phi)]
        # dNegLL/dF = em_w * [-d/(2*phi) + 0.5]  ... wait:
        # Let F = log(phi), phi = exp(F)
        # NegLL = em_w * [d/(2*exp(F)) + 0.5*F]
        # dNegLL/dF = em_w * [-d*exp(-F)/2 + 0.5] = em_w * [-d/(2*phi) + 0.5] = em_w * (0.5 - d/(2*phi))
        # d2NegLL/dF2 = em_w * d*exp(-F)/2 = em_w * d/(2*phi)  (positive)
        der1 = em_w * (0.5 - d / (2.0 * phi))
        der2 = np.maximum(em_w * d / (2.0 * phi), eps)

        return [(der1[i], der2[i]) for i in range(len(der1))]

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

    # For y > 0: first term y * (y^(1-p) - mu^(1-p)) / (1-p)
    # For y = 0: first term is 0
    t1 = np.where(
        y > 0,
        y * ((y ** p1) - (mu ** p1)) / p1,
        0.0,
    )

    # Second term: (y^(2-p) - mu^(2-p)) / (2-p)
    t2 = np.where(
        y > 0,
        ((y ** p2) - (mu ** p2)) / p2,
        -(mu ** p2) / p2,
    )

    deviance = 2.0 * (t1 - t2)
    deviance = np.maximum(deviance, 0.0)

    if exposure is not None:
        exposure_safe = np.maximum(exposure, eps)
        deviance = deviance / exposure_safe

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
            tweedie_zero_log = max(tweedie_zero_log, -700.0)
            ll[i] = np.log(q[i] + (1.0 - q[i]) * np.exp(tweedie_zero_log))
        else:
            tweedie_pos = (w / phi[i]) * (
                y[i] * (mu[i] ** p1) / p1 - (mu[i] ** p2) / p2
            )
            ll[i] = np.log(1.0 - q[i]) + tweedie_pos

    return ll
