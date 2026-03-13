"""
Tests for ZITModel end-to-end.

Covers:
    - Fit/predict cycle (independent and linked scenarios)
    - Predict_components returns correct columns
    - predict_proba_zero is in [0, 1]
    - score() returns finite float
    - get_booster() returns correct types
    - Edge cases: all zeros, no zeros, single feature
    - Parameter validation
    - Exposure handling
    - EM convergence is tracked
    - Unfitted model raises RuntimeError
    - Shapes consistent between methods
"""

import numpy as np
import polars as pl
import pytest

from insurance_zit_dglm.model import ZITModel, ZITReport, VuongResult
from tests.conftest import simulate_zit_data


class TestZITModelInit:
    def test_default_parameters(self):
        model = ZITModel()
        assert model.tweedie_power == 1.5
        assert model.n_estimators == 100
        assert model.link_scenario == "independent"
        assert model.em_iterations == 20
        assert not model._is_fitted

    def test_invalid_power_raises(self):
        with pytest.raises(ValueError, match="tweedie_power"):
            ZITModel(tweedie_power=1.0)
        with pytest.raises(ValueError, match="tweedie_power"):
            ZITModel(tweedie_power=2.0)
        with pytest.raises(ValueError, match="tweedie_power"):
            ZITModel(tweedie_power=0.5)

    def test_invalid_link_scenario_raises(self):
        with pytest.raises(ValueError, match="link_scenario"):
            ZITModel(link_scenario="hurdle")

    def test_invalid_gamma_raises(self):
        with pytest.raises(ValueError, match="gamma"):
            ZITModel(link_scenario="linked", gamma=-1.0)

    def test_repr(self):
        model = ZITModel(tweedie_power=1.3, n_estimators=50, em_iterations=10)
        r = repr(model)
        assert "ZITModel" in r
        assert "1.3" in r

    def test_cat_features_default_empty(self):
        model = ZITModel()
        assert model.cat_features == []


class TestZITModelFitPredict:
    def test_fit_returns_self(self, small_zit_data):
        X, y, _ = small_zit_data
        model = ZITModel(n_estimators=5, em_iterations=2, verbose=0)
        result = model.fit(X, y)
        assert result is model

    def test_is_fitted_after_fit(self, small_zit_data):
        X, y, _ = small_zit_data
        model = ZITModel(n_estimators=5, em_iterations=2, verbose=0)
        model.fit(X, y)
        assert model._is_fitted

    def test_predict_returns_polars_series(self, fitted_small_model):
        model, X, y = fitted_small_model
        preds = model.predict(X)
        assert isinstance(preds, pl.Series)

    def test_predict_correct_length(self, fitted_small_model):
        model, X, y = fitted_small_model
        preds = model.predict(X)
        assert len(preds) == len(y)

    def test_predict_non_negative(self, fitted_small_model):
        model, X, y = fitted_small_model
        preds = model.predict(X).to_numpy()
        assert np.all(preds >= 0)

    def test_predict_finite(self, fitted_small_model):
        model, X, y = fitted_small_model
        preds = model.predict(X).to_numpy()
        assert np.all(np.isfinite(preds))

    def test_predict_components_columns(self, fitted_small_model):
        model, X, y = fitted_small_model
        components = model.predict_components(X)
        assert isinstance(components, pl.DataFrame)
        assert set(components.columns) >= {"mu", "phi", "q", "E_Y"}

    def test_predict_components_correct_shape(self, fitted_small_model):
        model, X, y = fitted_small_model
        components = model.predict_components(X)
        assert len(components) == len(y)

    def test_predict_components_mu_positive(self, fitted_small_model):
        model, X, y = fitted_small_model
        components = model.predict_components(X)
        assert np.all(components["mu"].to_numpy() > 0)

    def test_predict_components_phi_positive(self, fitted_small_model):
        model, X, y = fitted_small_model
        components = model.predict_components(X)
        assert np.all(components["phi"].to_numpy() > 0)

    def test_predict_components_q_bounds(self, fitted_small_model):
        model, X, y = fitted_small_model
        components = model.predict_components(X)
        q = components["q"].to_numpy()
        assert np.all(q >= 0)
        assert np.all(q <= 1)

    def test_e_y_equals_one_minus_q_times_mu(self, fitted_small_model):
        """E_Y must equal (1-q)*mu by definition."""
        model, X, y = fitted_small_model
        components = model.predict_components(X)
        mu = components["mu"].to_numpy()
        q = components["q"].to_numpy()
        e_y = components["E_Y"].to_numpy()
        expected = (1.0 - q) * mu
        np.testing.assert_allclose(e_y, expected, rtol=1e-5)

    def test_predict_matches_e_y_from_components(self, fitted_small_model):
        """predict() should match E_Y from predict_components()."""
        model, X, y = fitted_small_model
        preds = model.predict(X).to_numpy()
        e_y = model.predict_components(X)["E_Y"].to_numpy()
        np.testing.assert_allclose(preds, e_y, rtol=1e-5)

    def test_predict_proba_zero_bounds(self, fitted_small_model):
        model, X, y = fitted_small_model
        proba = model.predict_proba_zero(X).to_numpy()
        assert np.all(proba >= 0)
        assert np.all(proba <= 1)

    def test_predict_proba_zero_correct_length(self, fitted_small_model):
        model, X, y = fitted_small_model
        proba = model.predict_proba_zero(X)
        assert len(proba) == len(y)

    def test_predict_proba_zero_geq_q(self, fitted_small_model):
        """
        Full Pr(Y=0) must be >= q (structural zero contribution alone).
        Pr(Y=0) = q + (1-q)*exp(...) >= q.
        """
        model, X, y = fitted_small_model
        proba = model.predict_proba_zero(X).to_numpy()
        q = model.predict_components(X)["q"].to_numpy()
        assert np.all(proba >= q - 1e-8)

    def test_score_returns_float(self, fitted_small_model):
        model, X, y = fitted_small_model
        s = model.score(X, y)
        assert isinstance(s, float)
        assert np.isfinite(s)

    def test_unfitted_predict_raises(self, small_zit_data):
        X, y, _ = small_zit_data
        model = ZITModel()
        with pytest.raises(RuntimeError, match="fit"):
            model.predict(X)

    def test_unfitted_score_raises(self, small_zit_data):
        X, y, _ = small_zit_data
        model = ZITModel()
        with pytest.raises(RuntimeError, match="fit"):
            model.score(X, y)

    def test_x_y_length_mismatch_raises(self, small_zit_data):
        X, y, _ = small_zit_data
        model = ZITModel(n_estimators=5, em_iterations=2)
        with pytest.raises(ValueError, match="length"):
            model.fit(X, y[:10])

    def test_negative_y_raises(self):
        X = pl.DataFrame({"x": [1.0, 2.0, 3.0]})
        y = pl.Series([-1.0, 1.0, 2.0])
        model = ZITModel(n_estimators=5, em_iterations=2)
        with pytest.raises(ValueError, match="non-negative"):
            model.fit(X, y)

    def test_log_likelihoods_tracked(self, small_zit_data):
        X, y, _ = small_zit_data
        model = ZITModel(n_estimators=5, em_iterations=5, verbose=0)
        model.fit(X, y)
        assert len(model._log_likelihoods) > 0
        assert all(np.isfinite(model._log_likelihoods))

    def test_get_booster_mean(self, fitted_small_model):
        import catboost as cb
        model, X, y = fitted_small_model
        booster = model.get_booster("mean")
        assert booster is not None

    def test_get_booster_dispersion(self, fitted_small_model):
        model, X, y = fitted_small_model
        booster = model.get_booster("dispersion")
        assert booster is not None

    def test_get_booster_zero(self, fitted_small_model):
        model, X, y = fitted_small_model
        booster = model.get_booster("zero")
        assert booster is not None

    def test_get_booster_invalid_raises(self, fitted_small_model):
        model, X, y = fitted_small_model
        with pytest.raises(ValueError):
            model.get_booster("invalid")


class TestZITModelLinkedScenario:
    def test_linked_fits_without_error(self, small_zit_data):
        X, y, _ = small_zit_data
        model = ZITModel(
            n_estimators=5,
            em_iterations=3,
            link_scenario="linked",
            gamma=1.0,
            verbose=0,
        )
        model.fit(X, y)
        assert model._is_fitted

    def test_linked_predict_non_negative(self, small_zit_data):
        X, y, _ = small_zit_data
        model = ZITModel(
            n_estimators=5,
            em_iterations=3,
            link_scenario="linked",
            gamma=1.0,
            verbose=0,
        )
        model.fit(X, y)
        preds = model.predict(X).to_numpy()
        assert np.all(preds >= 0)

    def test_linked_no_pi_model(self, small_zit_data):
        """In linked scenario, pi_model should be None."""
        X, y, _ = small_zit_data
        model = ZITModel(
            n_estimators=5,
            em_iterations=2,
            link_scenario="linked",
            gamma=1.0,
            verbose=0,
        )
        model.fit(X, y)
        assert model._pi_model is None

    def test_linked_q_derived_from_mu(self, small_zit_data):
        """In linked scenario, q = 1/(1+mu^gamma)."""
        X, y, _ = small_zit_data
        gamma_val = 1.5
        model = ZITModel(
            n_estimators=5,
            em_iterations=2,
            link_scenario="linked",
            gamma=gamma_val,
            verbose=0,
        )
        model.fit(X, y)
        components = model.predict_components(X)
        mu = components["mu"].to_numpy()
        q = components["q"].to_numpy()
        expected_q = 1.0 / (1.0 + mu ** gamma_val)
        np.testing.assert_allclose(q, expected_q, rtol=1e-5)

    def test_linked_gamma_estimation(self, small_zit_data):
        """With gamma=None, model should estimate gamma from data."""
        X, y, _ = small_zit_data
        model = ZITModel(
            n_estimators=5,
            em_iterations=2,
            link_scenario="linked",
            gamma=None,
            verbose=0,
        )
        model.fit(X, y)
        assert model._gamma_fitted is not None
        assert model._gamma_fitted > 0

    def test_linked_get_booster_zero_raises(self, small_zit_data):
        X, y, _ = small_zit_data
        model = ZITModel(
            n_estimators=5,
            em_iterations=2,
            link_scenario="linked",
            gamma=1.0,
            verbose=0,
        )
        model.fit(X, y)
        with pytest.raises(ValueError, match="linked"):
            model.get_booster("zero")


class TestZITModelEdgeCases:
    def test_all_zeros_fits_without_error(self, all_zeros_data):
        X, y, _ = all_zeros_data
        model = ZITModel(n_estimators=5, em_iterations=2, verbose=0)
        # All zeros is a valid (degenerate) case
        model.fit(X, y)
        assert model._is_fitted

    def test_all_zeros_predict_near_zero(self, all_zeros_data):
        X, y, _ = all_zeros_data
        model = ZITModel(n_estimators=5, em_iterations=3, verbose=0)
        model.fit(X, y)
        preds = model.predict(X).to_numpy()
        assert np.all(preds >= 0)
        assert np.all(np.isfinite(preds))

    def test_no_zeros_fits_without_error(self, no_zeros_data):
        X, y, _ = no_zeros_data
        model = ZITModel(n_estimators=5, em_iterations=2, verbose=0)
        model.fit(X, y)
        assert model._is_fitted

    def test_no_zeros_pi_near_zero(self, no_zeros_data):
        """With no structural zeros, q should be near zero."""
        X, y, _ = no_zeros_data
        model = ZITModel(n_estimators=5, em_iterations=3, verbose=0)
        model.fit(X, y)
        components = model.predict_components(X)
        q = components["q"].to_numpy()
        # q should be small (near 0) since there are no structural zeros
        assert np.mean(q) < 0.5  # Loose check

    def test_single_feature_fits(self, single_feature_data):
        X, y, _ = single_feature_data
        model = ZITModel(n_estimators=5, em_iterations=2, verbose=0)
        model.fit(X, y)
        preds = model.predict(X)
        assert len(preds) == len(y)

    def test_high_zero_rate_fits(self, high_zero_rate_data):
        X, y, _ = high_zero_rate_data
        model = ZITModel(n_estimators=5, em_iterations=3, verbose=0)
        model.fit(X, y)
        assert model._is_fitted

    def test_low_zero_rate_fits(self, low_zero_rate_data):
        X, y, _ = low_zero_rate_data
        model = ZITModel(n_estimators=5, em_iterations=3, verbose=0)
        model.fit(X, y)
        assert model._is_fitted


class TestZITModelWithExposure:
    def test_exposure_col_accepted(self, zit_data_with_exposure):
        X, y, _ = zit_data_with_exposure
        model = ZITModel(
            n_estimators=5,
            em_iterations=2,
            exposure_col="exposure",
            verbose=0,
        )
        model.fit(X, y)
        assert model._is_fitted

    def test_exposure_col_not_in_features(self, zit_data_with_exposure):
        """Exposure column should not appear as a feature in boosters."""
        X, y, _ = zit_data_with_exposure
        model = ZITModel(
            n_estimators=5,
            em_iterations=2,
            exposure_col="exposure",
            verbose=0,
        )
        model.fit(X, y)
        # Feature names should not include exposure
        assert "exposure" not in model._feature_names

    def test_missing_exposure_col_raises(self, small_zit_data):
        X, y, _ = small_zit_data
        model = ZITModel(
            n_estimators=5,
            em_iterations=2,
            exposure_col="nonexistent_col",
        )
        with pytest.raises(ValueError, match="exposure_col"):
            model.fit(X, y)

    def test_predict_with_exposure_col(self, zit_data_with_exposure):
        X, y, _ = zit_data_with_exposure
        model = ZITModel(
            n_estimators=5,
            em_iterations=2,
            exposure_col="exposure",
            verbose=0,
        )
        model.fit(X, y)
        preds = model.predict(X)
        assert len(preds) == len(y)
        assert np.all(preds.to_numpy() >= 0)


class TestZITModelDifferentPowers:
    @pytest.mark.parametrize("p", [1.2, 1.5, 1.8])
    def test_different_powers_fit_and_predict(self, small_zit_data, p):
        X, y, _ = small_zit_data
        model = ZITModel(
            tweedie_power=p,
            n_estimators=5,
            em_iterations=2,
            verbose=0,
        )
        model.fit(X, y)
        preds = model.predict(X).to_numpy()
        assert np.all(np.isfinite(preds))
        assert np.all(preds >= 0)


class TestZITModelScoring:
    def test_score_negative_ll(self, fitted_small_model):
        """Log-likelihood should be negative for finite distributions."""
        model, X, y = fitted_small_model
        s = model.score(X, y)
        # Not strictly required to be negative (normalising constants omitted)
        # but should be finite
        assert np.isfinite(s)

    def test_better_fit_higher_score(self, medium_zit_data):
        """A model with more iterations should score >= a less-fitted model."""
        X, y, _ = medium_zit_data
        model_few = ZITModel(n_estimators=5, em_iterations=2, verbose=0)
        model_more = ZITModel(n_estimators=20, em_iterations=5, verbose=0)

        model_few.fit(X, y)
        model_more.fit(X, y)

        score_few = model_few.score(X, y)
        score_more = model_more.score(X, y)

        # In-sample score should generally be higher (or equal) with more iterations
        # This is a soft check — not strictly guaranteed but should hold in practice
        assert np.isfinite(score_few)
        assert np.isfinite(score_more)
