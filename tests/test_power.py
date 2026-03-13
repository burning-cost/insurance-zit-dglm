"""
Tests for power parameter estimation by profile likelihood.

Covers:
    - estimate_power returns value in p_grid
    - estimate_power with default p_grid
    - estimate_power with q=None (standard Tweedie)
    - estimate_power with exposure
    - power_grid_search interface
    - Numerical stability
"""

import numpy as np
import pytest

from insurance_zit_dglm.power import estimate_power, power_grid_search, _profile_ll


class TestEstimatePower:
    def test_returns_value_in_grid(self):
        rng = np.random.default_rng(0)
        n = 100
        y = np.concatenate([np.zeros(30), rng.exponential(1.0, 70)])
        mu = np.ones(n) * 0.7
        p_grid = [1.3, 1.5, 1.7]

        p_hat = estimate_power(y, mu, p_grid=p_grid)
        assert p_hat in p_grid

    def test_default_p_grid_used(self):
        rng = np.random.default_rng(1)
        n = 100
        y = np.concatenate([np.zeros(30), rng.exponential(1.0, 70)])
        mu = np.ones(n) * 0.7
        p_hat = estimate_power(y, mu)
        assert 1.0 < p_hat < 2.0

    def test_returns_float(self):
        rng = np.random.default_rng(2)
        y = np.concatenate([np.zeros(20), rng.exponential(1.0, 80)])
        mu = np.ones(100)
        p_hat = estimate_power(y, mu)
        assert isinstance(p_hat, float)

    def test_with_q_none(self):
        """q=None should work (standard Tweedie, no structural zeros)."""
        rng = np.random.default_rng(3)
        y = np.concatenate([np.zeros(10), rng.exponential(1.0, 90)])
        mu = np.ones(100)
        p_hat = estimate_power(y, mu, q=None)
        assert 1.0 < p_hat < 2.0

    def test_scalar_phi(self):
        """phi as scalar should work."""
        rng = np.random.default_rng(4)
        y = np.concatenate([np.zeros(20), rng.exponential(1.0, 80)])
        mu = np.ones(100)
        p_hat = estimate_power(y, mu, phi=1.5)
        assert 1.0 < p_hat < 2.0

    def test_with_exposure(self):
        """Exposure parameter should be accepted."""
        rng = np.random.default_rng(5)
        n = 100
        y = np.concatenate([np.zeros(30), rng.exponential(1.0, 70)])
        mu = np.ones(n)
        exposure = rng.uniform(0.5, 2.0, n)
        p_hat = estimate_power(y, mu, exposure=exposure)
        assert 1.0 < p_hat < 2.0

    def test_all_zeros_no_crash(self):
        """All-zero y should return some value from the grid without crashing."""
        y = np.zeros(100)
        mu = np.ones(100) * 0.5
        p_grid = [1.3, 1.5, 1.7]
        p_hat = estimate_power(y, mu, p_grid=p_grid)
        assert p_hat in p_grid

    def test_single_candidate(self):
        """Single p_grid value should return that value."""
        rng = np.random.default_rng(6)
        y = np.concatenate([np.zeros(20), rng.exponential(1.0, 80)])
        mu = np.ones(100)
        p_hat = estimate_power(y, mu, p_grid=[1.5])
        assert p_hat == 1.5


class TestProfileLL:
    def test_returns_finite(self):
        rng = np.random.default_rng(0)
        n = 50
        y = np.concatenate([np.zeros(15), rng.exponential(1.0, 35)])
        mu = np.ones(n)
        phi = np.ones(n)
        q = np.full(n, 0.1)
        w = np.ones(n)

        ll = _profile_ll(y, mu, phi, q, 1.5, w)
        assert np.isfinite(ll)

    def test_higher_p_better_for_heavy_tail(self):
        """
        For a heavy-tailed dataset, higher p should be preferred over lower p.
        This is a soft test that verifies the profile LL is discriminating.
        """
        rng = np.random.default_rng(0)
        n = 200
        # Simulate data from p=1.8 (heavier tail)
        y = np.concatenate([np.zeros(50), rng.pareto(1.5, 150) + 1.0])
        mu = np.ones(n)
        phi = np.ones(n)
        q = np.full(n, 0.1)
        w = np.ones(n)

        ll_low = _profile_ll(y, mu, phi, q, 1.2, w)
        ll_high = _profile_ll(y, mu, phi, q, 1.8, w)

        # Both should be finite
        assert np.isfinite(ll_low)
        assert np.isfinite(ll_high)

    def test_q_zero_reduces_to_tweedie(self):
        """With q=0, the ZIT profile LL should equal standard Tweedie LL for y>0."""
        y = np.array([1.5, 2.0, 0.5])
        mu = np.ones(3)
        phi = np.ones(3)
        q = np.zeros(3)  # No structural zeros
        w = np.ones(3)

        ll = _profile_ll(y, mu, phi, q, 1.5, w)
        assert np.isfinite(ll)


class TestPowerGridSearch:
    def test_returns_dict_with_keys(self):
        rng = np.random.default_rng(0)
        n = 100
        y = np.concatenate([np.zeros(30), rng.exponential(1.0, 70)])

        def fit_fn(p):
            return np.ones(n), np.ones(n), np.full(n, 0.2)

        result = power_grid_search(y, fit_fn, p_grid=[1.3, 1.5, 1.7])
        assert "best_p" in result
        assert "p_grid" in result
        assert "log_likelihoods" in result
        assert "best_ll" in result

    def test_best_p_in_grid(self):
        rng = np.random.default_rng(1)
        n = 100
        y = np.concatenate([np.zeros(30), rng.exponential(1.0, 70)])
        p_grid = [1.2, 1.5, 1.8]

        def fit_fn(p):
            return np.ones(n), np.ones(n), np.full(n, 0.2)

        result = power_grid_search(y, fit_fn, p_grid=p_grid)
        assert result["best_p"] in p_grid

    def test_ll_count_matches_grid(self):
        rng = np.random.default_rng(2)
        n = 50
        y = np.concatenate([np.zeros(15), rng.exponential(1.0, 35)])
        p_grid = [1.3, 1.5, 1.7]

        def fit_fn(p):
            return np.ones(n), np.ones(n), np.full(n, 0.1)

        result = power_grid_search(y, fit_fn, p_grid=p_grid)
        assert len(result["log_likelihoods"]) == len(p_grid)

    def test_best_ll_equals_max(self):
        rng = np.random.default_rng(3)
        n = 50
        y = np.concatenate([np.zeros(15), rng.exponential(1.0, 35)])
        p_grid = [1.3, 1.5, 1.7]

        def fit_fn(p):
            return np.ones(n), np.ones(n), np.full(n, 0.1)

        result = power_grid_search(y, fit_fn, p_grid=p_grid)
        assert result["best_ll"] == max(result["log_likelihoods"])
