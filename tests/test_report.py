"""
Tests for ZITReport diagnostics and visualisation.

Covers:
    - calibration_plot returns a figure
    - zero_calibration_plot returns a figure
    - dispersion_plot returns a figure
    - lorenz_curve returns (figure, float)
    - Gini coefficient in [0, 1]
    - vuong_test returns VuongResult with correct fields
    - feature_importance returns a polars DataFrame
    - unfitted model raises ValueError
"""

import numpy as np
import polars as pl
import pytest

from insurance_zit_dglm.model import ZITModel, ZITReport, VuongResult
from tests.conftest import simulate_zit_data


@pytest.fixture
def report_fixture():
    X, y, _ = simulate_zit_data(n=300, seed=20)
    model = ZITModel(n_estimators=10, em_iterations=3, verbose=0)
    model.fit(X, y)
    report = ZITReport(model)
    return report, X, y


@pytest.fixture
def two_models_for_vuong():
    X, y, _ = simulate_zit_data(n=300, seed=30)
    model_1 = ZITModel(n_estimators=10, em_iterations=3, verbose=0)
    model_2 = ZITModel(n_estimators=5, em_iterations=2, verbose=0)
    model_1.fit(X, y)
    model_2.fit(X, y)
    return model_1, model_2, X, y


class TestZITReportInit:
    def test_unfitted_model_raises(self):
        model = ZITModel()
        with pytest.raises(ValueError, match="fitted"):
            ZITReport(model)

    def test_fitted_model_ok(self, report_fixture):
        report, X, y = report_fixture
        assert report.model._is_fitted


class TestCalibrationPlot:
    def test_returns_figure(self, report_fixture):
        import matplotlib
        matplotlib.use("Agg")
        report, X, y = report_fixture
        fig = report.calibration_plot(X, y, n_buckets=5)
        import matplotlib.pyplot as plt
        assert hasattr(fig, "savefig")
        plt.close("all")

    def test_custom_buckets(self, report_fixture):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        report, X, y = report_fixture
        fig = report.calibration_plot(X, y, n_buckets=3)
        assert fig is not None
        plt.close("all")

    def test_with_ax(self, report_fixture):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        report, X, y = report_fixture
        fig, ax = plt.subplots()
        returned_fig = report.calibration_plot(X, y, ax=ax)
        assert returned_fig is not None
        plt.close("all")


class TestZeroCalibrationPlot:
    def test_returns_figure(self, report_fixture):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        report, X, y = report_fixture
        fig = report.zero_calibration_plot(X, y)
        assert fig is not None
        plt.close("all")


class TestDispersionPlot:
    def test_returns_figure(self, report_fixture):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        report, X, y = report_fixture
        fig = report.dispersion_plot(X, y)
        assert fig is not None
        plt.close("all")


class TestLorenzCurve:
    def test_returns_figure_and_gini(self, report_fixture):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        report, X, y = report_fixture
        fig, gini = report.lorenz_curve(X, y)
        assert fig is not None
        assert isinstance(gini, float)
        plt.close("all")

    def test_gini_in_range(self, report_fixture):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        report, X, y = report_fixture
        _, gini = report.lorenz_curve(X, y)
        # Gini should be in [0, 1] for non-negative predictions
        assert -0.01 <= gini <= 1.01
        plt.close("all")

    def test_gini_finite(self, report_fixture):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        report, X, y = report_fixture
        _, gini = report.lorenz_curve(X, y)
        assert np.isfinite(gini)
        plt.close("all")


class TestVuongTest:
    def test_returns_vuong_result(self, report_fixture, two_models_for_vuong):
        report, _, _ = report_fixture
        model_1, model_2, X, y = two_models_for_vuong
        result = report.vuong_test(model_1, model_2, X, y)
        assert isinstance(result, VuongResult)

    def test_statistic_finite(self, report_fixture, two_models_for_vuong):
        report, _, _ = report_fixture
        model_1, model_2, X, y = two_models_for_vuong
        result = report.vuong_test(model_1, model_2, X, y)
        assert np.isfinite(result.statistic)

    def test_p_value_in_0_1(self, report_fixture, two_models_for_vuong):
        report, _, _ = report_fixture
        model_1, model_2, X, y = two_models_for_vuong
        result = report.vuong_test(model_1, model_2, X, y)
        assert 0.0 <= result.p_value <= 1.0

    def test_preferred_model_valid(self, report_fixture, two_models_for_vuong):
        report, _, _ = report_fixture
        model_1, model_2, X, y = two_models_for_vuong
        result = report.vuong_test(model_1, model_2, X, y)
        assert result.preferred_model in ("model_1", "model_2", "indeterminate")

    def test_n_observations_correct(self, report_fixture, two_models_for_vuong):
        report, _, _ = report_fixture
        model_1, model_2, X, y = two_models_for_vuong
        result = report.vuong_test(model_1, model_2, X, y)
        assert result.n_observations == len(y)

    def test_lr_shape_correct(self, report_fixture, two_models_for_vuong):
        report, _, _ = report_fixture
        model_1, model_2, X, y = two_models_for_vuong
        result = report.vuong_test(model_1, model_2, X, y)
        assert len(result.log_likelihood_ratios) == len(y)

    def test_identical_models_indeterminate(self, report_fixture):
        """Comparing a model against itself should be indeterminate."""
        import matplotlib
        matplotlib.use("Agg")
        report, X, y = report_fixture
        model = report.model
        result = report.vuong_test(model, model, X, y)
        assert result.preferred_model == "indeterminate"
        # LRs should be all zeros
        np.testing.assert_allclose(result.log_likelihood_ratios, 0.0, atol=1e-8)

    def test_antisymmetry(self, report_fixture, two_models_for_vuong):
        """Swapping model_1 and model_2 should negate the test statistic."""
        report, _, _ = report_fixture
        model_1, model_2, X, y = two_models_for_vuong
        result_12 = report.vuong_test(model_1, model_2, X, y)
        result_21 = report.vuong_test(model_2, model_1, X, y)
        assert abs(result_12.statistic + result_21.statistic) < 1e-6


class TestFeatureImportance:
    def test_returns_polars_dataframe(self, report_fixture):
        report, X, y = report_fixture
        fi = report.feature_importance("mean")
        assert isinstance(fi, pl.DataFrame)

    def test_has_feature_and_importance_columns(self, report_fixture):
        report, X, y = report_fixture
        fi = report.feature_importance("mean")
        assert "feature" in fi.columns
        assert "importance" in fi.columns

    def test_feature_count_matches_features(self, report_fixture):
        report, X, y = report_fixture
        n_features = len([c for c in X.columns])
        fi = report.feature_importance("mean")
        assert len(fi) == n_features

    def test_sorted_descending(self, report_fixture):
        report, X, y = report_fixture
        fi = report.feature_importance("mean")
        importances = fi["importance"].to_numpy()
        assert np.all(importances[:-1] >= importances[1:])

    def test_dispersion_head_importances(self, report_fixture):
        report, X, y = report_fixture
        fi = report.feature_importance("dispersion")
        assert len(fi) > 0

    def test_zero_head_importances(self, report_fixture):
        report, X, y = report_fixture
        fi = report.feature_importance("zero")
        assert len(fi) > 0

    def test_invalid_component_raises(self, report_fixture):
        report, X, y = report_fixture
        with pytest.raises(ValueError):
            report.feature_importance("invalid")

    def test_linked_zero_raises(self):
        X, y, _ = simulate_zit_data(n=100, seed=50)
        model = ZITModel(
            n_estimators=5,
            em_iterations=2,
            link_scenario="linked",
            gamma=1.0,
            verbose=0,
        )
        model.fit(X, y)
        report = ZITReport(model)
        with pytest.raises(ValueError, match="linked"):
            report.feature_importance("zero")
