"""
Tests for balance checking and recalibration.

Covers:
    - BalanceResult fields and repr
    - check_balance with balanced/imbalanced model
    - Group-level balance checks
    - recalibrate() creates RecalibratedZITModel
    - RecalibratedZITModel enforces global balance
    - Zero calibration ratio
    - Dispersion check
"""

import numpy as np
import polars as pl
import pytest

from insurance_zit_dglm.calibration import (
    check_balance,
    recalibrate,
    BalanceResult,
    RecalibratedZITModel,
)
from insurance_zit_dglm.model import ZITModel
from tests.conftest import simulate_zit_data


@pytest.fixture
def fitted_model_for_calibration():
    X, y, _ = simulate_zit_data(n=300, seed=10)
    model = ZITModel(n_estimators=10, em_iterations=3, verbose=0)
    model.fit(X, y)
    return model, X, y


class TestBalanceResult:
    def test_repr_balanced(self):
        result = BalanceResult(
            is_balanced=True,
            ratio=1.001,
            tolerance=0.02,
            n_observations=100,
            total_predicted=50.5,
            total_observed=50.4,
        )
        assert "BALANCED" in repr(result)
        assert "1.001" in repr(result) or "1.0" in repr(result)

    def test_repr_imbalanced(self):
        result = BalanceResult(
            is_balanced=False,
            ratio=0.80,
            tolerance=0.02,
            n_observations=100,
            total_predicted=40.0,
            total_observed=50.0,
        )
        assert "IMBALANCED" in repr(result)

    def test_is_balanced_within_tolerance(self):
        result = BalanceResult(
            is_balanced=True,
            ratio=0.999,
            tolerance=0.02,
            n_observations=500,
            total_predicted=100.0,
            total_observed=100.1,
        )
        assert result.is_balanced

    def test_is_balanced_outside_tolerance(self):
        result = BalanceResult(
            is_balanced=False,
            ratio=0.85,
            tolerance=0.02,
            n_observations=500,
            total_predicted=85.0,
            total_observed=100.0,
        )
        assert not result.is_balanced


class TestCheckBalance:
    def test_returns_balance_result(self, fitted_model_for_calibration):
        model, X, y = fitted_model_for_calibration
        result = check_balance(model, X, y)
        assert isinstance(result, BalanceResult)

    def test_ratio_finite(self, fitted_model_for_calibration):
        model, X, y = fitted_model_for_calibration
        result = check_balance(model, X, y)
        assert np.isfinite(result.ratio)

    def test_ratio_positive(self, fitted_model_for_calibration):
        model, X, y = fitted_model_for_calibration
        result = check_balance(model, X, y)
        assert result.ratio > 0

    def test_n_observations_correct(self, fitted_model_for_calibration):
        model, X, y = fitted_model_for_calibration
        result = check_balance(model, X, y)
        assert result.n_observations == len(y)

    def test_total_predicted_positive(self, fitted_model_for_calibration):
        model, X, y = fitted_model_for_calibration
        result = check_balance(model, X, y)
        assert result.total_predicted >= 0

    def test_total_observed_matches_y_sum(self, fitted_model_for_calibration):
        model, X, y = fitted_model_for_calibration
        result = check_balance(model, X, y)
        assert abs(result.total_observed - float(y.sum())) < 1e-6

    def test_zero_calibration_ratio_finite(self, fitted_model_for_calibration):
        model, X, y = fitted_model_for_calibration
        result = check_balance(model, X, y)
        assert np.isfinite(result.zero_calibration_ratio)

    def test_zero_calibration_ratio_positive(self, fitted_model_for_calibration):
        model, X, y = fitted_model_for_calibration
        result = check_balance(model, X, y)
        assert result.zero_calibration_ratio > 0

    def test_dispersion_check_finite(self, fitted_model_for_calibration):
        model, X, y = fitted_model_for_calibration
        result = check_balance(model, X, y)
        assert np.isfinite(result.dispersion_check)

    def test_group_results_when_groups_provided(self, fitted_model_for_calibration):
        model, X, y = fitted_model_for_calibration
        rng = np.random.default_rng(0)
        groups = pl.Series(rng.choice(["A", "B", "C"], size=len(y)))

        result = check_balance(model, X, y, groups=groups)
        assert len(result.group_results) == 3
        assert "A" in result.group_results
        assert "B" in result.group_results
        assert "C" in result.group_results

    def test_group_results_are_balance_results(self, fitted_model_for_calibration):
        model, X, y = fitted_model_for_calibration
        groups = pl.Series(np.where(np.arange(len(y)) < len(y) // 2, "low", "high"))
        result = check_balance(model, X, y, groups=groups)
        for grp_result in result.group_results.values():
            assert isinstance(grp_result, BalanceResult)

    def test_group_n_sums_to_total(self, fitted_model_for_calibration):
        model, X, y = fitted_model_for_calibration
        groups = pl.Series(["A"] * (len(y) // 2) + ["B"] * (len(y) - len(y) // 2))
        result = check_balance(model, X, y, groups=groups)
        total_n = sum(r.n_observations for r in result.group_results.values())
        assert total_n == len(y)

    def test_group_predicted_sums_to_total(self, fitted_model_for_calibration):
        model, X, y = fitted_model_for_calibration
        groups = pl.Series(["X"] * (len(y) // 3) + ["Y"] * (len(y) - len(y) // 3))
        result = check_balance(model, X, y, groups=groups)
        grp_sum = sum(r.total_predicted for r in result.group_results.values())
        assert abs(grp_sum - result.total_predicted) < 1e-4

    def test_custom_tolerance(self, fitted_model_for_calibration):
        model, X, y = fitted_model_for_calibration
        result_strict = check_balance(model, X, y, tolerance=0.001)
        result_loose = check_balance(model, X, y, tolerance=0.99)
        # With very loose tolerance, should always be balanced
        assert result_loose.is_balanced

    def test_all_zeros_y_no_error(self):
        X, y_raw, _ = simulate_zit_data(n=100, seed=4, all_zeros=True)
        y = y_raw
        model = ZITModel(n_estimators=5, em_iterations=2, verbose=0)
        model.fit(X, y)
        result = check_balance(model, X, y)
        assert isinstance(result, BalanceResult)


class TestRecalibrate:
    def test_returns_recalibrated_model(self, fitted_model_for_calibration):
        model, X, y = fitted_model_for_calibration
        recal = recalibrate(model, X, y)
        assert isinstance(recal, RecalibratedZITModel)

    def test_correction_factor_finite(self, fitted_model_for_calibration):
        model, X, y = fitted_model_for_calibration
        recal = recalibrate(model, X, y)
        assert np.isfinite(recal.correction_factor)
        assert recal.correction_factor > 0

    def test_recalibrated_sum_equals_observed(self, fitted_model_for_calibration):
        """After recalibration, sum of predictions should equal sum of observed."""
        model, X, y = fitted_model_for_calibration
        recal = recalibrate(model, X, y)
        preds = recal.predict(X).to_numpy()
        y_np = y.to_numpy()
        # Recalibrated on the same dataset, so should be exact
        assert abs(np.sum(preds) - np.sum(y_np)) < 1e-4

    def test_recalibrated_predict_non_negative(self, fitted_model_for_calibration):
        model, X, y = fitted_model_for_calibration
        recal = recalibrate(model, X, y)
        preds = recal.predict(X).to_numpy()
        assert np.all(preds >= 0)

    def test_recalibrated_predict_components_has_e_y(self, fitted_model_for_calibration):
        model, X, y = fitted_model_for_calibration
        recal = recalibrate(model, X, y)
        components = recal.predict_components(X)
        assert "E_Y" in components.columns

    def test_recalibrated_components_e_y_scaled(self, fitted_model_for_calibration):
        model, X, y = fitted_model_for_calibration
        recal = recalibrate(model, X, y)
        raw_e_y = model.predict(X).to_numpy()
        recal_e_y = recal.predict(X).to_numpy()
        ratio = recal_e_y / np.maximum(raw_e_y, 1e-10)
        # All ratios should equal correction_factor
        np.testing.assert_allclose(ratio, recal.correction_factor, rtol=1e-5)

    def test_recalibrated_score_finite(self, fitted_model_for_calibration):
        model, X, y = fitted_model_for_calibration
        recal = recalibrate(model, X, y)
        s = recal.score(X, y)
        assert np.isfinite(s)
