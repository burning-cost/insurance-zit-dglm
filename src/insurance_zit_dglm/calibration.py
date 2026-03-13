"""
Autocalibration and balance property checks for ZIT-DGLM.

Implements the Delong-Wuthrich (arXiv:2103.03635) balance property check
adapted for zero-inflated models, where the effective mean is E[Y] = (1-q)*mu.

The balance property requires:
    sum_i E[Y_i | x_i] = sum_i y_i

This is NOT guaranteed by gradient boosting minimising ZIT deviance. The check
and optional recalibration step are essential for UK actuarial practice, where
regulatory fair pricing obligations (FCA Consumer Duty, PRA Technical Provisions)
require demonstrably balanced premium predictions.

Reference: Delong & Wuthrich (2021), arXiv:2103.03635, Insurance: Mathematics
and Economics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import polars as pl


@dataclass
class BalanceResult:
    """
    Result from a balance property check on a ZIT-DGLM.

    Attributes
    ----------
    is_balanced:
        True if sum(E_Y) / sum(y) is within tolerance of 1.0.
    ratio:
        sum(E_Y) / sum(y). Values above 1 indicate systematic over-prediction.
    tolerance:
        Tolerance used to determine is_balanced.
    n_observations:
        Number of observations checked.
    total_predicted:
        sum of predicted E[Y_i].
    total_observed:
        sum of observed y_i.
    group_results:
        Balance results per group (if groups provided). Dict mapping group
        value to sub-BalanceResult for that group.
    zero_calibration_ratio:
        mean(Pr(Y=0)) / observed_zero_rate. Close to 1 = well-calibrated zeros.
    dispersion_check:
        mean(D(y;mu)/phi) for non-zero observations. Should be ~1 under correct specification.
    """

    is_balanced: bool
    ratio: float
    tolerance: float
    n_observations: int
    total_predicted: float
    total_observed: float
    group_results: dict = field(default_factory=dict)
    zero_calibration_ratio: float = 1.0
    dispersion_check: float = 1.0

    def __repr__(self) -> str:
        status = "BALANCED" if self.is_balanced else "IMBALANCED"
        return (
            f"BalanceResult({status}, ratio={self.ratio:.4f}, "
            f"n={self.n_observations})"
        )


def check_balance(
    model: "ZITModel",
    X: pl.DataFrame,
    y: pl.Series,
    groups: Optional[pl.Series] = None,
    tolerance: float = 0.02,
) -> BalanceResult:
    """
    Check the balance property for a fitted ZIT-DGLM.

    The balance property requires sum(E[Y_i]) ~ sum(y_i). Deviation beyond
    the tolerance threshold indicates systematic bias in the model.

    Also checks zero calibration and dispersion.

    Parameters
    ----------
    model:
        A fitted ZITModel.
    X:
        Feature DataFrame.
    y:
        Observed aggregate losses.
    groups:
        Optional categorical series for group-level balance checks.
        For example, age band, vehicle group, or region.
    tolerance:
        Acceptable relative deviation from balance. Default 2%.

    Returns
    -------
    BalanceResult

    Examples
    --------
    >>> result = check_balance(model, X_test, y_test)
    >>> print(result.ratio)
    0.9987
    >>> print(result.is_balanced)
    True
    """
    from insurance_zit_dglm.losses import tweedie_unit_deviance

    y_np = y.to_numpy().astype(float)
    n = len(y_np)

    components = model.predict_components(X)
    e_y = components["E_Y"].to_numpy()
    mu = components["mu"].to_numpy()
    phi = components["phi"].to_numpy()

    total_pred = float(np.sum(e_y))
    total_obs = float(np.sum(y_np))

    ratio = total_pred / max(total_obs, 1e-10)
    is_balanced = abs(ratio - 1.0) <= tolerance

    # Zero calibration
    pred_zero_prob = model.predict_proba_zero(X).to_numpy()
    obs_zero_rate = float(np.mean(y_np <= 0.0))
    pred_zero_rate = float(np.mean(pred_zero_prob))
    zero_cal_ratio = pred_zero_rate / max(obs_zero_rate, 1e-10)

    # Dispersion check: mean scaled deviance for non-zero observations
    pos_mask = y_np > 0.0
    if np.any(pos_mask):
        unit_dev = tweedie_unit_deviance(y_np[pos_mask], mu[pos_mask], model.tweedie_power)
        disp_check = float(np.mean(unit_dev / np.maximum(phi[pos_mask], 1e-10)))
    else:
        disp_check = float("nan")

    # Group-level checks
    group_results: dict = {}
    if groups is not None:
        groups_np = groups.to_numpy()
        unique_groups = np.unique(groups_np)

        for grp in unique_groups:
            mask = groups_np == grp
            grp_pred = float(np.sum(e_y[mask]))
            grp_obs = float(np.sum(y_np[mask]))
            grp_ratio = grp_pred / max(grp_obs, 1e-10)
            grp_balanced = abs(grp_ratio - 1.0) <= tolerance

            group_results[grp] = BalanceResult(
                is_balanced=grp_balanced,
                ratio=grp_ratio,
                tolerance=tolerance,
                n_observations=int(np.sum(mask)),
                total_predicted=grp_pred,
                total_observed=grp_obs,
            )

    return BalanceResult(
        is_balanced=is_balanced,
        ratio=ratio,
        tolerance=tolerance,
        n_observations=n,
        total_predicted=total_pred,
        total_observed=total_obs,
        group_results=group_results,
        zero_calibration_ratio=zero_cal_ratio,
        dispersion_check=disp_check,
    )


def recalibrate(
    model: "ZITModel",
    X: pl.DataFrame,
    y: pl.Series,
) -> "RecalibratedZITModel":
    """
    Apply the Delong-Wuthrich extra GLM recalibration step to a ZIT-DGLM.

    Fits a simple intercept correction on log(E[Y]) to enforce the balance
    property. This is a lightweight post-hoc fix: the underlying model is
    unchanged; only the predicted values are rescaled.

    The recalibration model:
        log(E[Y_recal]) = log(E[Y_raw]) + alpha_0

    where alpha_0 is estimated by MLE on the recalibration dataset.

    Parameters
    ----------
    model:
        A fitted ZITModel.
    X:
        Recalibration feature DataFrame (typically held-out validation set).
    y:
        Observed aggregate losses on recalibration set.

    Returns
    -------
    RecalibratedZITModel
        Wrapper that applies the intercept correction at prediction time.
    """
    y_np = y.to_numpy().astype(float)
    e_y_raw = model.predict(X).to_numpy()

    total_obs = float(np.sum(y_np))
    total_pred = float(np.sum(e_y_raw))

    # Multiplicative intercept correction
    correction_factor = total_obs / max(total_pred, 1e-10)

    return RecalibratedZITModel(model=model, correction_factor=correction_factor)


class RecalibratedZITModel:
    """
    Wrapper around a ZITModel that applies a multiplicative intercept correction.

    The correction_factor = sum(y) / sum(E[Y_raw]) ensures global balance
    on the recalibration dataset.

    Parameters
    ----------
    model:
        A fitted ZITModel.
    correction_factor:
        Multiplicative correction to apply to E[Y] predictions.
    """

    def __init__(self, model: "ZITModel", correction_factor: float) -> None:
        self.model = model
        self.correction_factor = correction_factor

    def predict(self, X: pl.DataFrame) -> pl.Series:
        """Return recalibrated E[Y] predictions."""
        raw = self.model.predict(X).to_numpy()
        return pl.Series(raw * self.correction_factor)

    def predict_components(self, X: pl.DataFrame) -> pl.DataFrame:
        """Return recalibrated components (correction applied to E_Y only)."""
        components = self.model.predict_components(X)
        corrected_e_y = components["E_Y"] * self.correction_factor
        return components.with_columns(corrected_e_y.alias("E_Y"))

    def score(self, X: pl.DataFrame, y: pl.Series) -> float:
        """Score based on recalibrated predictions."""
        from insurance_zit_dglm.losses import zit_log_likelihood

        y_np = y.to_numpy().astype(float)
        data = self.model._prepare_data(X, y)
        mu, phi, q = self.model._predict_components_np(
            data.X_np, data.cat_feature_indices
        )
        # Recalibrate mu: E[Y]_recal = (1-q)*mu*factor => mu_recal = mu*factor
        mu_recal = mu * self.correction_factor
        ll = zit_log_likelihood(
            y_np, mu_recal, phi, q, self.model.tweedie_power, data.exposure_np
        )
        return float(np.mean(ll))
