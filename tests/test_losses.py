"""
Tests for custom CatBoost loss functions and utility calculations.

Covers:
    - ZITTweedieLoss gradient/Hessian correctness for y=0 and y>0
    - ZITZeroInflationLoss gradient/Hessian correctness
    - ZITDispersionLoss gradient/Hessian correctness
    - tweedie_unit_deviance correctness and edge cases
    - zit_log_likelihood correctness and boundary conditions
    - Numerical consistency between gradient and finite differences
"""

import numpy as np
import pytest

from insurance_zit_dglm.losses import (
    ZITTweedieLoss,
    ZITZeroInflationLoss,
    ZITDispersionLoss,
    tweedie_unit_deviance,
    zit_log_likelihood,
    _zit_tweedie_ders,
)


# ---------------------------------------------------------------------------
# Helper: finite difference gradient check
# ---------------------------------------------------------------------------


def fd_gradient(f, x, eps=1e-5):
    """Central finite difference approximation of df/dx."""
    return (f(x + eps) - f(x - eps)) / (2 * eps)


# ---------------------------------------------------------------------------
# ZITTweedieLoss
# ---------------------------------------------------------------------------


class TestZITTweedieLoss:
    def _make_loss(self, n=5, p=1.5):
        rng = np.random.default_rng(0)
        phi = rng.uniform(0.5, 2.0, n)
        q = rng.uniform(0.05, 0.4, n)
        em_w = rng.uniform(0.5, 1.0, n)
        exposure = np.ones(n)
        return ZITTweedieLoss(p, phi, q, em_w, exposure)

    def test_returns_list_of_tuples(self):
        loss = self._make_loss(5)
        approxes = [0.1, -0.2, 0.3, -0.1, 0.0]
        targets = [0.0, 1.5, 0.0, 2.0, 0.0]
        result = loss.calc_ders_range(approxes, targets, None)
        assert len(result) == 5
        for g, h in result:
            assert isinstance(g, (int, float, np.floating))
            assert isinstance(h, (int, float, np.floating))

    def test_hessian_positive(self):
        """Hessian (negative of returned value) should be positive for stability."""
        loss = self._make_loss(10)
        rng = np.random.default_rng(1)
        approxes = list(rng.uniform(-1, 1, 10))
        targets = list(np.where(rng.random(10) < 0.3, 0.0, rng.exponential(1.0, 10)))
        result = loss.calc_ders_range(approxes, targets, None)
        for _, h in result:
            # CatBoost returns -H, so h should be >= 0 (positive curvature)
            assert -h >= 0 or abs(h) < 1e-8  # h is negative of hessian, so >=0

    def test_zero_observations_different_from_positive(self):
        """y=0 and y>0 observations should produce different gradient patterns."""
        n = 2
        phi = np.array([1.0, 1.0])
        q = np.array([0.2, 0.2])
        em_w = np.array([1.0, 1.0])
        exposure = np.ones(2)
        loss = ZITTweedieLoss(1.5, phi, q, em_w, exposure)

        result_zero = loss.calc_ders_range([0.0], [0.0], None)
        result_pos = loss.calc_ders_range([0.0], [1.0], None)
        # Gradients should differ between y=0 and y>0
        assert result_zero[0][0] != result_pos[0][0]

    def test_em_weight_scales_gradient(self):
        """EM weight should scale gradient and hessian linearly."""
        p = 1.5
        phi = np.array([1.0])
        q = np.array([0.1])
        em_w_full = np.array([1.0])
        em_w_half = np.array([0.5])
        exposure = np.ones(1)

        loss_full = ZITTweedieLoss(p, phi, q, em_w_full, exposure)
        loss_half = ZITTweedieLoss(p, phi, q, em_w_half, exposure)

        g_full, h_full = loss_full.calc_ders_range([0.0], [0.0], None)[0]
        g_half, h_half = loss_half.calc_ders_range([0.0], [0.0], None)[0]

        assert abs(g_full / g_half - 2.0) < 1e-6
        assert abs(h_full / h_half - 2.0) < 1e-6

    def test_is_max_optimal_false(self):
        loss = self._make_loss()
        assert not loss.is_max_optimal()

    def test_gradient_finite_difference_y0(self):
        """Check gradient agrees with numerical finite difference for y=0."""
        p = 1.5
        phi = np.array([1.2])
        q = np.array([0.2])
        em_w = np.array([1.0])
        exposure = np.ones(1)

        from insurance_zit_dglm.losses import _zit_tweedie_ders

        F = 0.3  # log-scale score

        def neg_ll(F_val):
            mu = np.exp(F_val)
            # ZIT log-likelihood for y=0
            p2 = 2.0 - p
            log_tz = -(mu ** p2) / (phi[0] * p2)
            log_tz = max(log_tz, -700)
            return np.log(q[0] + (1 - q[0]) * np.exp(log_tz))

        # FD gradient of log-lik w.r.t. F
        fd_g = fd_gradient(neg_ll, F)

        g, h = _zit_tweedie_ders(
            mu=np.exp(F), y=0.0, p=p,
            phi=phi[0], q=q[0], em_weight=1.0, exposure=1.0
        )
        # Our g is negative gradient of neg_ll (i.e., positive gradient of ll)
        # fd_g is gradient of ll w.r.t. F
        assert abs((-g) - fd_g) < 5e-4

    def test_gradient_finite_difference_y_positive(self):
        """Check gradient agrees with numerical finite difference for y>0."""
        p = 1.5
        phi_val = 1.2
        q_val = 0.0  # no structural zeros for positive y
        F = 0.3

        def ll_pos(F_val):
            mu = np.exp(F_val)
            p1 = 1.0 - p
            p2 = 2.0 - p
            y_val = 2.0
            return (1.0 / phi_val) * (y_val * (mu ** p1) / p1 - (mu ** p2) / p2)

        fd_g = fd_gradient(ll_pos, F)

        g, h = _zit_tweedie_ders(
            mu=np.exp(F), y=2.0, p=p,
            phi=phi_val, q=q_val, em_weight=1.0, exposure=1.0
        )
        assert abs((-g) - fd_g) < 5e-4

    def test_power_values(self):
        """Loss should work for different power values in (1, 2)."""
        for p in [1.1, 1.3, 1.5, 1.7, 1.9]:
            phi = np.array([1.0])
            q = np.array([0.1])
            em_w = np.array([1.0])
            exposure = np.ones(1)
            loss = ZITTweedieLoss(p, phi, q, em_w, exposure)
            result = loss.calc_ders_range([0.0], [1.0], None)
            g, h = result[0]
            assert np.isfinite(g)
            assert np.isfinite(h)


# ---------------------------------------------------------------------------
# ZITZeroInflationLoss
# ---------------------------------------------------------------------------


class TestZITZeroInflationLoss:
    def test_returns_correct_shape(self):
        pi_em = np.array([0.0, 0.3, 0.0, 0.8, 0.0])
        loss = ZITZeroInflationLoss(pi_em)
        result = loss.calc_ders_range([0.1, -0.5, 0.2, -1.0, 0.0], [0, 0, 0, 0, 0], None)
        assert len(result) == 5

    def test_gradient_at_pi_equals_label(self):
        """When q = Pi_i, gradient should be ~0."""
        # Pi_i = 0.5, q = 0.5 => gradient = q - Pi = 0
        pi_em = np.array([0.5])
        loss = ZITZeroInflationLoss(pi_em)
        # logit(0.5) = 0
        result = loss.calc_ders_range([0.0], [0.0], None)
        g, h = result[0]
        # -g = q - Pi = 0.5 - 0.5 = 0
        assert abs(g) < 1e-6

    def test_hessian_sigmoid_form(self):
        """Hessian should equal q*(1-q)."""
        pi_em = np.array([0.3])
        loss = ZITZeroInflationLoss(pi_em)
        F = 0.5
        q = 1.0 / (1.0 + np.exp(-F))
        result = loss.calc_ders_range([F], [0.0], None)
        _, h = result[0]
        expected_hess = q * (1.0 - q)
        assert abs(-h - expected_hess) < 1e-6

    def test_gradient_agrees_fd(self):
        """Gradient agrees with finite difference of binary cross-entropy."""
        Pi = 0.4
        pi_em = np.array([Pi])
        loss = ZITZeroInflationLoss(pi_em)
        F = 0.3

        def bce(F_val):
            q = 1.0 / (1.0 + np.exp(-F_val))
            return Pi * np.log(q + 1e-15) + (1 - Pi) * np.log(1 - q + 1e-15)

        fd_g = fd_gradient(bce, F)
        g, _ = loss.calc_ders_range([F], [0.0], None)
        assert abs((-g) - fd_g) < 5e-4

    def test_positive_obs_zero_pi(self):
        """For y>0, Pi=0 means gradient = q (model should push q down)."""
        pi_em = np.array([0.0])  # Pi=0 for positive obs
        loss = ZITZeroInflationLoss(pi_em)
        F = 0.0  # q = 0.5
        result = loss.calc_ders_range([F], [1.0], None)
        g, _ = result[0]
        # gradient = q - Pi = 0.5 - 0 = 0.5; returned as -g = -0.5
        assert abs(g - (-0.5)) < 1e-6

    def test_is_max_optimal_false(self):
        loss = ZITZeroInflationLoss(np.zeros(1))
        assert not loss.is_max_optimal()


# ---------------------------------------------------------------------------
# ZITDispersionLoss
# ---------------------------------------------------------------------------


class TestZITDispersionLoss:
    def test_returns_correct_shape(self):
        rng = np.random.default_rng(0)
        n = 8
        d = rng.uniform(0.1, 5.0, n)
        em_w = np.ones(n)
        loss = ZITDispersionLoss(d, em_w)
        result = loss.calc_ders_range(list(np.zeros(n)), list(d), None)
        assert len(result) == n

    def test_em_weight_zero_gives_zero_gradient(self):
        """EM weight = 0 (observation is certainly structural zero) => zero gradient."""
        d = np.array([2.0])
        em_w = np.array([0.0])
        loss = ZITDispersionLoss(d, em_w)
        result = loss.calc_ders_range([0.0], [2.0], None)
        g, _ = result[0]
        assert abs(g) < 1e-9

    def test_gradient_stationary_at_expected_deviance(self):
        """Gradient = 0 when phi = d/2 (expected deviance from chi^2(1))."""
        # At stationarity: d/(2*phi) = 0.5 => phi = d
        # Actually l = em_w * [-d/(2*phi) - 0.5*log(phi)]
        # dl/dphi = em_w * [d/(2*phi^2) - 1/(2*phi)] = em_w/(2*phi) * [d/phi - 1]
        # dl/dF = phi * dl/dphi = em_w/2 * [d/phi - 1]
        # = 0 when d = phi
        d_val = 2.0
        phi_val = d_val  # stationary point
        F_val = np.log(phi_val)
        em_w = np.array([1.0])
        d = np.array([d_val])
        loss = ZITDispersionLoss(d, em_w)
        result = loss.calc_ders_range([F_val], [d_val], None)
        g, _ = result[0]
        # gradient of ll at stationary point should be 0
        assert abs(g) < 1e-6

    def test_is_max_optimal_false(self):
        loss = ZITDispersionLoss(np.ones(1), np.ones(1))
        assert not loss.is_max_optimal()


# ---------------------------------------------------------------------------
# tweedie_unit_deviance
# ---------------------------------------------------------------------------


class TestTweedieUnitDeviance:
    def test_zero_deviance_when_y_equals_mu(self):
        """Unit deviance D(y; mu) = 0 when y = mu."""
        mu = np.array([1.0, 2.0, 0.5])
        y = mu.copy()
        d = tweedie_unit_deviance(y, mu, 1.5)
        np.testing.assert_allclose(d, 0.0, atol=1e-10)

    def test_non_negative(self):
        """Deviance is always non-negative."""
        rng = np.random.default_rng(0)
        y = np.concatenate([np.zeros(20), rng.exponential(1.0, 80)])
        mu = rng.uniform(0.1, 3.0, 100)
        d = tweedie_unit_deviance(y, mu, 1.5)
        assert np.all(d >= 0)

    def test_y_zero_special_case(self):
        """D(0; mu) = 2*mu^(2-p)/(2-p) for y=0."""
        mu = np.array([1.0, 2.0])
        p = 1.5
        d = tweedie_unit_deviance(np.zeros(2), mu, p)
        expected = 2.0 * (mu ** (2.0 - p)) / (2.0 - p)
        np.testing.assert_allclose(d, expected, rtol=1e-6)

    def test_exposure_scaling(self):
        """With exposure w, result should be D/w (scaled deviance)."""
        y = np.array([1.5])
        mu = np.array([1.0])
        exposure = np.array([2.0])
        d_no_exp = tweedie_unit_deviance(y, mu, 1.5)
        d_with_exp = tweedie_unit_deviance(y, mu, 1.5, exposure)
        np.testing.assert_allclose(d_with_exp, d_no_exp / 2.0, rtol=1e-6)

    def test_different_power_values(self):
        """Deviance should be finite and positive for all valid powers."""
        y = np.array([1.5])
        mu = np.array([1.0])
        for p in [1.1, 1.3, 1.5, 1.7, 1.9]:
            d = tweedie_unit_deviance(y, mu, p)
            assert np.all(np.isfinite(d))
            assert np.all(d > 0)

    def test_larger_y_gives_larger_deviance(self):
        """Deviance should increase as y moves further from mu."""
        mu = np.array([1.0])
        p = 1.5
        d1 = tweedie_unit_deviance(np.array([1.5]), mu, p)
        d2 = tweedie_unit_deviance(np.array([3.0]), mu, p)
        assert d2 > d1


# ---------------------------------------------------------------------------
# zit_log_likelihood
# ---------------------------------------------------------------------------


class TestZITLogLikelihood:
    def test_y_zero_uses_mixture_formula(self):
        """For y=0, ll = log(q + (1-q)*exp(-mu^p2/(phi*p2)))."""
        y = np.array([0.0])
        mu = np.array([1.0])
        phi = np.array([1.0])
        q = np.array([0.3])
        p = 1.5

        ll = zit_log_likelihood(y, mu, phi, q, p)
        p2 = 2.0 - p
        expected = np.log(0.3 + 0.7 * np.exp(-(1.0 ** p2) / p2))
        np.testing.assert_allclose(ll[0], expected, rtol=1e-6)

    def test_y_positive_excludes_structural_zeros(self):
        """For y>0, log(1-q) + Tweedie_ll."""
        y = np.array([1.5])
        mu = np.array([1.0])
        phi = np.array([1.0])
        q = np.array([0.2])
        p = 1.5

        ll = zit_log_likelihood(y, mu, phi, q, p)
        p1 = 1.0 - p
        p2 = 2.0 - p
        tweedie_part = (1.5 * (1.0 ** p1) / p1 - (1.0 ** p2) / p2)
        expected = np.log(0.8) + tweedie_part
        np.testing.assert_allclose(ll[0], expected, rtol=1e-6)

    def test_q_zero_reduces_to_standard_tweedie_y0(self):
        """With q=0, ZIT reduces to standard Tweedie for y=0."""
        y = np.array([0.0])
        mu = np.array([0.5])
        phi = np.array([1.2])
        q = np.array([0.0])
        p = 1.5

        ll = zit_log_likelihood(y, mu, phi, q, p)
        # Standard Tweedie zero: log(exp(-mu^(2-p)/(phi*(2-p)))) = -mu^(2-p)/(phi*(2-p))
        p2 = 2.0 - p
        expected = -mu[0] ** p2 / (phi[0] * p2)
        np.testing.assert_allclose(ll[0], expected, rtol=1e-6)

    def test_exposure_affects_zero_probability(self):
        """Higher exposure should reduce Tweedie zero probability, changing ll for y=0."""
        y = np.array([0.0])
        mu = np.array([0.5])
        phi = np.array([1.0])
        q = np.array([0.1])
        p = 1.5

        ll_e1 = zit_log_likelihood(y, mu, phi, q, p, np.array([1.0]))
        ll_e5 = zit_log_likelihood(y, mu, phi, q, p, np.array([5.0]))

        # Larger exposure => smaller Tweedie zero prob => smaller probability of observing 0
        # So ll should differ
        assert ll_e1[0] != ll_e5[0]

    def test_all_finite(self):
        """Log-likelihood should be finite for reasonable inputs."""
        rng = np.random.default_rng(99)
        n = 50
        y = np.concatenate([np.zeros(20), rng.exponential(1.0, 30)])
        mu = rng.uniform(0.1, 3.0, n)
        phi = rng.uniform(0.5, 2.0, n)
        q = rng.uniform(0.01, 0.5, n)

        ll = zit_log_likelihood(y, mu, phi, q, 1.5)
        assert np.all(np.isfinite(ll))

    def test_ll_all_zeros(self):
        """Log-likelihood should handle all-zero y gracefully."""
        n = 20
        y = np.zeros(n)
        mu = np.ones(n)
        phi = np.ones(n)
        q = np.full(n, 0.5)
        ll = zit_log_likelihood(y, mu, phi, q, 1.5)
        assert np.all(np.isfinite(ll))
        assert np.all(ll <= 0)  # probabilities <= 1 => log-ll <= 0

    def test_ll_all_positive(self):
        """Log-likelihood should handle all-positive y gracefully."""
        rng = np.random.default_rng(77)
        n = 20
        y = rng.exponential(1.0, n) + 0.01  # strictly positive
        mu = np.ones(n)
        phi = np.ones(n)
        q = np.full(n, 0.01)
        ll = zit_log_likelihood(y, mu, phi, q, 1.5)
        assert np.all(np.isfinite(ll))
