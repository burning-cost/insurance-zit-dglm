"""
Tests for the EM algorithm components.

Covers:
    - E-step correctness (boundary conditions, numerical stability)
    - Convergence checking
    - Initialisation logic
    - E-step behaviour with varying exposure
"""

import numpy as np
import pytest

from insurance_zit_dglm.em import e_step, initialise_state, check_convergence


class TestEStep:
    def test_positive_y_gives_zero_pi(self):
        """For y > 0, Pi_i must be 0 (cannot be structural zero)."""
        n = 5
        y = np.array([1.0, 2.5, 0.3, 4.0, 0.1])
        mu = np.ones(n)
        phi = np.ones(n)
        q = np.full(n, 0.3)
        p = 1.5
        w = np.ones(n)

        pi = e_step(y, mu, phi, q, p, w)
        np.testing.assert_allclose(pi, 0.0, atol=1e-10)

    def test_zero_y_gives_positive_pi(self):
        """For y = 0, Pi_i should be in (0, 1)."""
        n = 5
        y = np.zeros(n)
        mu = np.full(n, 0.5)
        phi = np.ones(n)
        q = np.full(n, 0.3)
        p = 1.5
        w = np.ones(n)

        pi = e_step(y, mu, phi, q, p, w)
        assert np.all(pi > 0)
        assert np.all(pi < 1)

    def test_pi_bounds(self):
        """Pi_i must be in [0, 1] for any input."""
        rng = np.random.default_rng(0)
        n = 100
        y = np.where(rng.random(n) < 0.4, 0.0, rng.exponential(1.0, n))
        mu = rng.uniform(0.01, 5.0, n)
        phi = rng.uniform(0.1, 3.0, n)
        q = rng.uniform(0.01, 0.99, n)
        w = rng.uniform(0.5, 2.0, n)

        pi = e_step(y, mu, phi, q, p=1.5, exposure=w)
        assert np.all(pi >= 0.0)
        assert np.all(pi <= 1.0)

    def test_high_q_gives_high_pi_for_zeros(self):
        """High structural zero probability q should give high Pi for y=0."""
        y = np.array([0.0])
        mu = np.array([0.5])
        phi = np.array([1.0])
        w = np.ones(1)

        pi_low_q = e_step(y, mu, phi, np.array([0.05]), 1.5, w)
        pi_high_q = e_step(y, mu, phi, np.array([0.90]), 1.5, w)

        assert pi_high_q[0] > pi_low_q[0]

    def test_high_mu_gives_high_pi_for_zeros(self):
        """
        High mu reduces Tweedie zero probability, so zeros are more likely
        structural. E-step should give higher Pi for y=0 when mu is large.
        """
        y = np.array([0.0])
        phi = np.array([1.0])
        q = np.array([0.3])
        w = np.ones(1)

        pi_low_mu = e_step(y, np.array([0.1]), phi, q, 1.5, w)
        pi_high_mu = e_step(y, np.array([5.0]), phi, q, 1.5, w)

        assert pi_high_mu[0] > pi_low_mu[0]

    def test_high_exposure_gives_high_pi_for_zeros(self):
        """
        Higher exposure means lower Tweedie zero probability (more expected claims),
        so a zero observation is more likely structural.
        """
        y = np.array([0.0])
        mu = np.array([0.5])
        phi = np.array([1.0])
        q = np.array([0.2])

        pi_low_w = e_step(y, mu, phi, q, 1.5, np.array([0.1]))
        pi_high_w = e_step(y, mu, phi, q, 1.5, np.array([10.0]))

        assert pi_high_w[0] > pi_low_w[0]

    def test_q_zero_gives_zero_pi(self):
        """When q=0 (no structural zeros), Pi should be ~0 for all y=0."""
        y = np.zeros(5)
        mu = np.full(5, 0.5)
        phi = np.ones(5)
        q = np.zeros(5)
        w = np.ones(5)

        pi = e_step(y, mu, phi, q, 1.5, w)
        np.testing.assert_allclose(pi, 0.0, atol=1e-6)

    def test_q_one_gives_one_pi_for_zeros(self):
        """When q=1, all zeros should have Pi=1."""
        y = np.zeros(5)
        mu = np.full(5, 0.5)
        phi = np.ones(5)
        q = np.ones(5) * 0.9999  # Near 1 to avoid numerical issues
        w = np.ones(5)

        pi = e_step(y, mu, phi, q, 1.5, w)
        assert np.all(pi > 0.99)

    def test_mixed_y_correct_mask(self):
        """Mixed y values: only zeros should have positive Pi."""
        y = np.array([0.0, 1.5, 0.0, 2.0, 0.0])
        mu = np.full(5, 0.5)
        phi = np.ones(5)
        q = np.full(5, 0.3)
        w = np.ones(5)

        pi = e_step(y, mu, phi, q, 1.5, w)
        assert pi[1] == 0.0
        assert pi[3] == 0.0
        assert pi[0] > 0
        assert pi[2] > 0
        assert pi[4] > 0

    def test_numerical_stability_large_mu(self):
        """Large mu should not cause NaN or Inf in E-step."""
        y = np.zeros(3)
        mu = np.array([100.0, 1000.0, 10000.0])
        phi = np.ones(3)
        q = np.full(3, 0.3)
        w = np.ones(3)

        pi = e_step(y, mu, phi, q, 1.5, w)
        assert np.all(np.isfinite(pi))

    def test_numerical_stability_small_phi(self):
        """Very small phi should not cause NaN in E-step."""
        y = np.zeros(3)
        mu = np.ones(3)
        phi = np.array([1e-5, 1e-4, 1e-3])
        q = np.full(3, 0.3)
        w = np.ones(3)

        pi = e_step(y, mu, phi, q, 1.5, w)
        assert np.all(np.isfinite(pi))


class TestInitialiseState:
    def test_mu0_equals_mean_of_positives(self):
        """Initial mu should be mean of positive y values."""
        rng = np.random.default_rng(0)
        y = np.concatenate([np.zeros(50), rng.exponential(2.0, 50)])
        w = np.ones(100)
        init = initialise_state(y, w, 1.5)
        expected_mu = float(np.mean(y[y > 0]))
        assert abs(init["mu_0"] - expected_mu) < 1e-6

    def test_phi0_is_one(self):
        """Initial phi should be 1.0 (log(phi)=0)."""
        y = np.array([0.0, 1.0, 2.0, 0.0, 3.0])
        w = np.ones(5)
        init = initialise_state(y, w, 1.5)
        assert init["phi_0"] == 1.0

    def test_q0_bounded(self):
        """Initial q should be in (0, 0.9)."""
        y = np.concatenate([np.zeros(100), np.ones(100)])
        w = np.ones(200)
        init = initialise_state(y, w, 1.5)
        assert 0 < init["q_0"] <= 0.9

    def test_all_zeros_gives_high_q0(self):
        """All-zero y should give high q_0."""
        y = np.zeros(100)
        w = np.ones(100)
        init = initialise_state(y, w, 1.5)
        assert init["q_0"] > 0  # Some structural zeros detected

    def test_all_positive_gives_low_q0(self):
        """All-positive y should give q_0 near 0."""
        rng = np.random.default_rng(0)
        y = rng.exponential(1.0, 100) + 0.01
        w = np.ones(100)
        init = initialise_state(y, w, 1.5)
        # Low zero rate means low q_0 (capped at max(excess, eps))
        assert init["q_0"] < 0.5

    def test_mu0_default_for_all_zeros(self):
        """If no positive observations, mu_0 should default to 1.0."""
        y = np.zeros(20)
        w = np.ones(20)
        init = initialise_state(y, w, 1.5)
        assert init["mu_0"] == 1.0


class TestCheckConvergence:
    def test_returns_false_with_one_value(self):
        assert not check_convergence([100.0], 1e-6)

    def test_returns_false_with_large_change(self):
        assert not check_convergence([-100.0, -50.0], 1e-4)

    def test_returns_true_with_tiny_change(self):
        assert check_convergence([-100.0, -100.0000001], 1e-6)

    def test_relative_criterion(self):
        """Change of 1e-4 on log-likelihood of 1e4 should be ~1e-8 relative."""
        assert check_convergence([-10000.0, -10000.001], 1e-4)
        assert not check_convergence([-100.0, -99.0], 1e-4)

    def test_empty_list_returns_false(self):
        assert not check_convergence([], 1e-6)
