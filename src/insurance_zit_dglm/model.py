"""
ZITModel: Zero-Inflated Tweedie Double GLM with CatBoost gradient boosting.

Three CatBoost models in an EM loop:
    mu_model:  CatBoostRegressor, log link, ZIT Tweedie custom loss
    phi_model: CatBoostRegressor, log link, gamma pseudo-likelihood
    pi_model:  CatBoostClassifier, logit link, EM-weighted logistic loss

EM algorithm (Gu arXiv:2405.14990 Algorithm 1):
    E-step: compute posterior P(structural zero | y_i=0, x_i)
    M-step: update mu, then phi, then pi models with EM-weighted objectives

The linked scenario (link_scenario='linked') uses a single CatBoost tree
for the mu head, with q derived as q = 1/(1+mu^gamma), following So & Valdez
arXiv:2406.16206 Scenario 2.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import polars as pl

try:
    import catboost as cb
except ImportError as e:
    raise ImportError(
        "catboost is required. Install with: pip install catboost"
    ) from e

from insurance_zit_dglm.losses import (
    ZITTweedieLoss,
    ZITZeroInflationLoss,
    ZITDispersionLoss,
    tweedie_unit_deviance,
    zit_log_likelihood,
)
from insurance_zit_dglm.em import e_step, initialise_state, check_convergence


@dataclass
class VuongResult:
    """Result from the Vuong test comparing two nested or non-nested models."""

    statistic: float
    p_value: float
    preferred_model: str  # 'model_1', 'model_2', or 'indeterminate'
    n_observations: int
    log_likelihood_ratios: np.ndarray


@dataclass
class _FitData:
    """Internal storage of training arrays extracted from Polars input."""

    X_np: np.ndarray
    y_np: np.ndarray
    exposure_np: np.ndarray
    feature_names: list[str]
    cat_feature_indices: list[int]
    n: int


class ZITModel:
    """
    Zero-Inflated Tweedie Double GLM with CatBoost gradient boosting.

    The ZIT distribution is a mixture:
        f(y) = q * I(y=0) + (1-q) * Tweedie(mu, phi, p)

    where the Tweedie component is compound Poisson-Gamma with power p in (1, 2).

    Expected value: E[Y] = (1-q) * mu

    Three separate CatBoost models are fitted inside an EM loop:
        - mu model (mean, log link): ZIT Tweedie custom objective
        - phi model (dispersion, log link): gamma pseudo-likelihood
        - pi model (zero-inflation probability, logit link): EM-weighted logistic

    The EM algorithm propagates uncertainty about which zero observations are
    structural zeros (z_i=1) versus Tweedie zeros (z_i=0, N_i=0).

    Parameters
    ----------
    tweedie_power:
        Tweedie power p in (1, 2). Compound Poisson-Gamma regime.
        Use estimate_power() to select this from data.
    n_estimators:
        Number of boosting rounds per EM iteration per head.
    learning_rate:
        CatBoost learning rate, applied to all three heads.
    max_depth:
        Tree depth for all three heads.
    em_iterations:
        Maximum outer EM loop iterations.
    em_tol:
        Convergence tolerance on relative log-likelihood change.
    link_scenario:
        'independent': three separate models (default, Gu/Scenario 1).
        'linked': single tree determines mu; q = 1/(1+mu^gamma) (Scenario 2).
    gamma:
        Exponent in q = 1/(1+mu^gamma) for linked scenario.
        If None, estimated by grid search during fit().
    cat_features:
        Names of categorical features. Passed to CatBoost.
    exposure_col:
        Column name for exposure weights w_i in the input DataFrame.
        Exposure enters the Tweedie deviance AND the E-step zero probability.
    verbose:
        CatBoost verbosity. Set to 0 to suppress output.
    random_seed:
        Random seed for reproducibility.

    Examples
    --------
    >>> import polars as pl
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> n = 200
    >>> X = pl.DataFrame({"x1": rng.normal(size=n), "x2": rng.normal(size=n)})
    >>> y = pl.Series(np.where(rng.random(n) < 0.3, 0.0, rng.exponential(1.0, n)))
    >>> model = ZITModel(n_estimators=10, em_iterations=3)
    >>> model.fit(X, y)
    ZITModel(tweedie_power=1.5, n_estimators=10, em_iterations=3)
    >>> preds = model.predict(X)
    >>> len(preds) == n
    True
    """

    def __init__(
        self,
        tweedie_power: float = 1.5,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 6,
        em_iterations: int = 20,
        em_tol: float = 1e-6,
        link_scenario: str = "independent",
        gamma: Optional[float] = None,
        cat_features: list[str] | None = None,
        exposure_col: Optional[str] = None,
        verbose: int = 0,
        random_seed: int = 42,
    ) -> None:
        if tweedie_power <= 1.0 or tweedie_power >= 2.0:
            raise ValueError(
                f"tweedie_power must be in (1, 2) for compound Poisson-Gamma; got {tweedie_power}"
            )
        if link_scenario not in ("independent", "linked"):
            raise ValueError(
                f"link_scenario must be 'independent' or 'linked'; got {link_scenario!r}"
            )
        if link_scenario == "linked" and gamma is not None and gamma <= 0:
            raise ValueError(f"gamma must be positive; got {gamma}")

        self.tweedie_power = tweedie_power
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.em_iterations = em_iterations
        self.em_tol = em_tol
        self.link_scenario = link_scenario
        self.gamma = gamma
        self.cat_features = cat_features or []
        self.exposure_col = exposure_col
        self.verbose = verbose
        self.random_seed = random_seed

        # Fitted components — set during fit()
        self._mu_model: Optional[cb.CatBoostRegressor] = None
        self._phi_model: Optional[cb.CatBoostRegressor] = None
        self._pi_model: Optional[cb.CatBoostClassifier] = None
        self._gamma_fitted: Optional[float] = None
        self._log_likelihoods: list[float] = []
        self._feature_names: list[str] = []
        self._is_fitted: bool = False

    def __repr__(self) -> str:
        return (
            f"ZITModel(tweedie_power={self.tweedie_power}, "
            f"n_estimators={self.n_estimators}, "
            f"em_iterations={self.em_iterations})"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X: pl.DataFrame, y: pl.Series) -> "ZITModel":
        """
        Fit the ZIT-DGLM model via EM algorithm.

        Parameters
        ----------
        X:
            Feature DataFrame. Must include exposure_col if specified.
        y:
            Observed aggregate losses (>= 0). Non-negative, semi-continuous.

        Returns
        -------
        ZITModel
            Fitted model (self).
        """
        data = self._prepare_data(X, y)

        if self.link_scenario == "linked":
            self._fit_linked(data)
        else:
            self._fit_independent(data)

        self._is_fitted = True
        return self

    def predict(self, X: pl.DataFrame) -> pl.Series:
        """
        Predict expected aggregate loss E[Y|x] = (1 - q(x)) * mu(x).

        Parameters
        ----------
        X:
            Feature DataFrame.

        Returns
        -------
        polars.Series
            Predicted expected values, same length as X.
        """
        self._check_fitted()
        components = self.predict_components(X)
        return components["E_Y"]

    def predict_components(self, X: pl.DataFrame) -> pl.DataFrame:
        """
        Predict all ZIT components: mu, phi, q, and E[Y].

        Parameters
        ----------
        X:
            Feature DataFrame.

        Returns
        -------
        polars.DataFrame
            Columns: mu, phi, q, E_Y.
            - mu: Tweedie mean (conditional on non-structural-zero)
            - phi: Tweedie dispersion
            - q: structural zero probability
            - E_Y: expected aggregate loss = (1-q)*mu
        """
        self._check_fitted()
        X_np, _, _, cat_indices = self._extract_features(X)

        mu, phi, q = self._predict_components_np(X_np, cat_indices)
        e_y = (1.0 - q) * mu

        return pl.DataFrame(
            {
                "mu": mu,
                "phi": phi,
                "q": q,
                "E_Y": e_y,
            }
        )

    def predict_proba_zero(self, X: pl.DataFrame) -> pl.Series:
        """
        Predict the full probability of observing zero: Pr(Y=0|x).

        Pr(Y=0|x) = q(x) + (1-q(x)) * exp(-mu(x)^(2-p) / (phi(x)*(2-p)))

        Includes both structural zeros (probability q) and compound Poisson zeros
        (probability of no claims from the Tweedie component).

        Parameters
        ----------
        X:
            Feature DataFrame.

        Returns
        -------
        polars.Series
            Zero probabilities in [0, 1].
        """
        self._check_fitted()
        X_np, exposure_np, _, cat_indices = self._extract_features(X)

        mu, phi, q = self._predict_components_np(X_np, cat_indices)
        p2 = 2.0 - self.tweedie_power

        log_tweedie_zero = -(mu ** p2) / (phi * p2)
        log_tweedie_zero = np.maximum(log_tweedie_zero, -700.0)
        tweedie_zero = np.exp(log_tweedie_zero)
        prob_zero = q + (1.0 - q) * tweedie_zero

        return pl.Series(prob_zero)

    def score(self, X: pl.DataFrame, y: pl.Series) -> float:
        """
        Compute mean ZIT log-likelihood (higher is better).

        Parameters
        ----------
        X:
            Feature DataFrame.
        y:
            Observed aggregate losses.

        Returns
        -------
        float
            Mean ZIT log-likelihood.
        """
        self._check_fitted()
        data = self._prepare_data(X, y)

        mu, phi, q = self._predict_components_np(data.X_np, data.cat_feature_indices)
        ll = zit_log_likelihood(
            data.y_np, mu, phi, q, self.tweedie_power, data.exposure_np
        )
        return float(np.mean(ll))

    def get_booster(self, component: str) -> cb.CatBoostRegressor | cb.CatBoostClassifier:
        """
        Return the fitted CatBoost model for the specified component.

        Parameters
        ----------
        component:
            One of 'mean', 'dispersion', or 'zero'.

        Returns
        -------
        catboost.CatBoostRegressor or catboost.CatBoostClassifier
        """
        self._check_fitted()
        mapping = {
            "mean": self._mu_model,
            "dispersion": self._phi_model,
            "zero": self._pi_model,
        }
        if component not in mapping:
            raise ValueError(
                f"component must be one of {list(mapping.keys())}; got {component!r}"
            )
        return mapping[component]

    # ------------------------------------------------------------------
    # Fitting implementations
    # ------------------------------------------------------------------

    def _fit_independent(self, data: _FitData) -> None:
        """
        Fit the independent (Scenario 1) ZIT-DGLM via three-head EM loop.
        """
        y = data.y_np
        w = data.exposure_np
        n = data.n
        p = self.tweedie_power

        # Initialise parameter arrays
        init = initialise_state(y, w, p)
        mu = np.full(n, init["mu_0"])
        phi = np.full(n, init["phi_0"])
        q = np.full(n, init["q_0"])

        # Build initial CatBoost pools (feature data only, targets updated each iteration)
        cat_idxs = data.cat_feature_indices
        X_pool = self._make_pool(data.X_np, np.zeros(n), cat_idxs)

        log_likelihoods: list[float] = []

        for em_iter in range(self.em_iterations):
            # E-step
            pi_em = e_step(y, mu, phi, q, p, w)
            em_weights = 1.0 - pi_em  # weight for Tweedie component

            # Compute observed log-likelihood
            ll_vals = zit_log_likelihood(y, mu, phi, q, p, w)
            ll_total = float(np.sum(ll_vals))
            log_likelihoods.append(ll_total)

            if check_convergence(log_likelihoods, self.em_tol):
                break

            # M-step 1: update mu model (ZIT Tweedie loss)
            mu_loss = ZITTweedieLoss(p, phi, q, em_weights, w)
            mu_pool = self._make_pool(data.X_np, np.log(np.maximum(mu, 1e-10)), cat_idxs)
            mu_train_pool = cb.Pool(
                data=data.X_np,
                label=y,
                cat_features=cat_idxs,
            )

            if self._mu_model is None:
                self._mu_model = cb.CatBoostRegressor(
                    iterations=self.n_estimators,
                    learning_rate=self.learning_rate,
                    depth=self.max_depth,
                    loss_function=mu_loss,
                    verbose=self.verbose,
                    random_seed=self.random_seed,
                    eval_metric='RMSE',
                    allow_writing_files=False,
                )
                self._mu_model.fit(mu_train_pool)
            else:
                # Continue training with updated loss
                new_mu_model = cb.CatBoostRegressor(
                    iterations=self.n_estimators,
                    learning_rate=self.learning_rate,
                    depth=self.max_depth,
                    loss_function=mu_loss,
                    verbose=self.verbose,
                    random_seed=self.random_seed,
                    eval_metric='RMSE',
                    allow_writing_files=False,
                )
                new_mu_model.fit(mu_train_pool, init_model=self._mu_model)
                self._mu_model = new_mu_model

            mu_raw = self._mu_model.predict(mu_train_pool)
            mu = np.exp(mu_raw)
            mu = np.maximum(mu, 1e-10)

            # M-step 2: update pi model (zero-inflation, EM-weighted logistic)
            # Use y as binary label (whether obs is structural zero)
            # EM soft labels: Pi_i for y=0, 0 for y>0
            pi_labels = np.where(y <= 0.0, np.ones(n), np.zeros(n))
            pi_weights = np.abs(pi_em - pi_labels) + 1e-6  # ensure non-zero weights
            pi_weights = np.where(y <= 0.0, pi_em + 1e-6, em_weights + 1e-6)

            pi_loss = ZITZeroInflationLoss(pi_em)
            pi_train_pool = cb.Pool(
                data=data.X_np,
                label=pi_labels.astype(np.float32),
                weight=pi_weights,
                cat_features=cat_idxs,
            )

            if self._pi_model is None:
                self._pi_model = cb.CatBoostRegressor(
                    iterations=self.n_estimators,
                    learning_rate=self.learning_rate,
                    depth=self.max_depth,
                    loss_function=pi_loss,
                    verbose=self.verbose,
                    random_seed=self.random_seed,
                    eval_metric='RMSE',
                    allow_writing_files=False,
                )
                self._pi_model.fit(pi_train_pool)
            else:
                new_pi_model = cb.CatBoostRegressor(
                    iterations=self.n_estimators,
                    learning_rate=self.learning_rate,
                    depth=self.max_depth,
                    loss_function=pi_loss,
                    verbose=self.verbose,
                    random_seed=self.random_seed,
                    eval_metric='RMSE',
                    allow_writing_files=False,
                )
                new_pi_model.fit(pi_train_pool, init_model=self._pi_model)
                self._pi_model = new_pi_model

            pi_raw = self._pi_model.predict(pi_train_pool)
            q = 1.0 / (1.0 + np.exp(-pi_raw))
            q = np.clip(q, 1e-10, 1.0 - 1e-10)

            # M-step 3: update phi model (gamma pseudo-likelihood)
            unit_dev = tweedie_unit_deviance(y, mu, p, w)
            unit_dev = np.maximum(unit_dev, 1e-10)

            phi_loss = ZITDispersionLoss(unit_dev, em_weights)
            phi_train_pool = cb.Pool(
                data=data.X_np,
                label=unit_dev,
                weight=em_weights,
                cat_features=cat_idxs,
            )

            if self._phi_model is None:
                self._phi_model = cb.CatBoostRegressor(
                    iterations=self.n_estimators,
                    learning_rate=self.learning_rate,
                    depth=self.max_depth,
                    loss_function=phi_loss,
                    verbose=self.verbose,
                    random_seed=self.random_seed,
                    eval_metric='RMSE',
                    allow_writing_files=False,
                )
                self._phi_model.fit(phi_train_pool)
            else:
                new_phi_model = cb.CatBoostRegressor(
                    iterations=self.n_estimators,
                    learning_rate=self.learning_rate,
                    depth=self.max_depth,
                    loss_function=phi_loss,
                    verbose=self.verbose,
                    random_seed=self.random_seed,
                    eval_metric='RMSE',
                    allow_writing_files=False,
                )
                new_phi_model.fit(phi_train_pool, init_model=self._phi_model)
                self._phi_model = new_phi_model

            phi_raw = self._phi_model.predict(phi_train_pool)
            phi = np.exp(phi_raw)
            phi = np.maximum(phi, 1e-10)

        self._log_likelihoods = log_likelihoods
        self._feature_names = data.feature_names

    def _fit_linked(self, data: _FitData) -> None:
        """
        Fit the linked (Scenario 2) ZIT-DGLM where q = 1/(1 + mu^gamma).

        A single mu model determines both the mean and the structural zero probability
        through the functional link q = 1/(1 + exp(gamma * F_mu)).

        If gamma is None, estimate it by grid search over [0.1, 0.2, ..., 2.0].
        """
        y = data.y_np
        w = data.exposure_np
        n = data.n
        p = self.tweedie_power
        cat_idxs = data.cat_feature_indices

        # Estimate gamma if not specified
        if self.gamma is None:
            self._gamma_fitted = self._estimate_gamma(data)
        else:
            self._gamma_fitted = self.gamma

        gamma = self._gamma_fitted
        init = initialise_state(y, w, p)
        mu = np.full(n, init["mu_0"])
        phi = np.full(n, init["phi_0"])

        # In linked scenario: q = 1/(1+mu^gamma)
        q = 1.0 / (1.0 + mu ** gamma)

        log_likelihoods: list[float] = []

        for em_iter in range(self.em_iterations):
            pi_em = e_step(y, mu, phi, q, p, w)
            em_weights = 1.0 - pi_em

            ll_vals = zit_log_likelihood(y, mu, phi, q, p, w)
            ll_total = float(np.sum(ll_vals))
            log_likelihoods.append(ll_total)

            if check_convergence(log_likelihoods, self.em_tol):
                break

            # M-step 1: update mu model; q derived from mu
            mu_loss = ZITTweedieLoss(p, phi, q, em_weights, w)
            mu_train_pool = cb.Pool(
                data=data.X_np,
                label=y,
                cat_features=cat_idxs,
            )

            if self._mu_model is None:
                self._mu_model = cb.CatBoostRegressor(
                    iterations=self.n_estimators,
                    learning_rate=self.learning_rate,
                    depth=self.max_depth,
                    loss_function=mu_loss,
                    verbose=self.verbose,
                    random_seed=self.random_seed,
                    eval_metric='RMSE',
                    allow_writing_files=False,
                )
                self._mu_model.fit(mu_train_pool)
            else:
                new_mu_model = cb.CatBoostRegressor(
                    iterations=self.n_estimators,
                    learning_rate=self.learning_rate,
                    depth=self.max_depth,
                    loss_function=mu_loss,
                    verbose=self.verbose,
                    random_seed=self.random_seed,
                    eval_metric='RMSE',
                    allow_writing_files=False,
                )
                new_mu_model.fit(mu_train_pool, init_model=self._mu_model)
                self._mu_model = new_mu_model

            mu_raw = self._mu_model.predict(mu_train_pool)
            mu = np.exp(mu_raw)
            mu = np.maximum(mu, 1e-10)

            # Linked scenario: q from mu
            q = 1.0 / (1.0 + mu ** gamma)

            # M-step 2: update phi model
            unit_dev = tweedie_unit_deviance(y, mu, p, w)
            unit_dev = np.maximum(unit_dev, 1e-10)

            phi_loss = ZITDispersionLoss(unit_dev, em_weights)
            phi_train_pool = cb.Pool(
                data=data.X_np,
                label=unit_dev,
                weight=em_weights,
                cat_features=cat_idxs,
            )

            if self._phi_model is None:
                self._phi_model = cb.CatBoostRegressor(
                    iterations=self.n_estimators,
                    learning_rate=self.learning_rate,
                    depth=self.max_depth,
                    loss_function=phi_loss,
                    verbose=self.verbose,
                    random_seed=self.random_seed,
                    eval_metric='RMSE',
                    allow_writing_files=False,
                )
                self._phi_model.fit(phi_train_pool)
            else:
                new_phi_model = cb.CatBoostRegressor(
                    iterations=self.n_estimators,
                    learning_rate=self.learning_rate,
                    depth=self.max_depth,
                    loss_function=phi_loss,
                    verbose=self.verbose,
                    random_seed=self.random_seed,
                    eval_metric='RMSE',
                    allow_writing_files=False,
                )
                new_phi_model.fit(phi_train_pool, init_model=self._phi_model)
                self._phi_model = new_phi_model

            phi_raw = self._phi_model.predict(phi_train_pool)
            phi = np.exp(phi_raw)
            phi = np.maximum(phi, 1e-10)

        # Linked scenario has no separate pi model
        self._pi_model = None
        self._log_likelihoods = log_likelihoods
        self._feature_names = data.feature_names

    def _estimate_gamma(self, data: _FitData) -> float:
        """
        Estimate gamma by grid search: fit linked model at each candidate,
        return the value with highest log-likelihood.
        """
        gamma_grid = [round(0.1 * k, 1) for k in range(1, 21)]  # 0.1 to 2.0
        best_gamma = 1.0
        best_ll = -np.inf

        y = data.y_np
        w = data.exposure_np
        n = data.n
        p = self.tweedie_power
        cat_idxs = data.cat_feature_indices

        for gamma in gamma_grid:
            # Quick fit: 2 EM iterations to estimate log-likelihood
            init = initialise_state(y, w, p)
            mu = np.full(n, init["mu_0"])
            phi = np.full(n, init["phi_0"])
            q = 1.0 / (1.0 + mu ** gamma)

            pi_em = e_step(y, mu, phi, q, p, w)
            em_weights = 1.0 - pi_em

            ll_vals = zit_log_likelihood(y, mu, phi, q, p, w)
            ll = float(np.sum(ll_vals))

            if ll > best_ll:
                best_ll = ll
                best_gamma = gamma

        return best_gamma

    # ------------------------------------------------------------------
    # Prediction helpers
    # ------------------------------------------------------------------

    def _predict_components_np(
        self,
        X_np: np.ndarray,
        cat_idxs: list[int],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (mu, phi, q) as numpy arrays."""
        pool = cb.Pool(data=X_np, cat_features=cat_idxs)

        mu_raw = self._mu_model.predict(pool)
        mu = np.exp(mu_raw)
        mu = np.maximum(mu, 1e-10)

        phi_raw = self._phi_model.predict(pool)
        phi = np.exp(phi_raw)
        phi = np.maximum(phi, 1e-10)

        if self.link_scenario == "linked":
            gamma = self._gamma_fitted
            q = 1.0 / (1.0 + mu ** gamma)
        else:
            pi_raw = self._pi_model.predict(pool)
            q = 1.0 / (1.0 + np.exp(-pi_raw))
            q = np.clip(q, 1e-10, 1.0 - 1e-10)

        return mu, phi, q

    # ------------------------------------------------------------------
    # Data preparation
    # ------------------------------------------------------------------

    def _prepare_data(self, X: pl.DataFrame, y: pl.Series) -> _FitData:
        """Extract features, exposure, and labels from Polars inputs."""
        if len(X) != len(y):
            raise ValueError(
                f"X and y must have the same length; got {len(X)} and {len(y)}"
            )

        feature_cols = [c for c in X.columns if c != self.exposure_col]

        if self.exposure_col is not None:
            if self.exposure_col not in X.columns:
                raise ValueError(
                    f"exposure_col {self.exposure_col!r} not found in X columns"
                )
            exposure_np = X[self.exposure_col].to_numpy().astype(float)
        else:
            exposure_np = np.ones(len(y), dtype=float)

        exposure_np = np.maximum(exposure_np, 1e-10)

        X_features = X.select(feature_cols)
        X_np = X_features.to_numpy()
        y_np = y.to_numpy().astype(float)

        if np.any(y_np < 0):
            raise ValueError("y must be non-negative (aggregate losses >= 0)")

        cat_feature_indices = [
            i for i, col in enumerate(feature_cols) if col in self.cat_features
        ]

        return _FitData(
            X_np=X_np,
            y_np=y_np,
            exposure_np=exposure_np,
            feature_names=feature_cols,
            cat_feature_indices=cat_feature_indices,
            n=len(y_np),
        )

    def _extract_features(
        self, X: pl.DataFrame
    ) -> tuple[np.ndarray, np.ndarray, list[str], list[int]]:
        """Extract feature array for prediction."""
        feature_cols = [c for c in X.columns if c != self.exposure_col]

        if self.exposure_col is not None and self.exposure_col in X.columns:
            exposure_np = X[self.exposure_col].to_numpy().astype(float)
        else:
            exposure_np = np.ones(len(X), dtype=float)

        X_features = X.select(feature_cols)
        X_np = X_features.to_numpy()

        cat_feature_indices = [
            i for i, col in enumerate(feature_cols) if col in self.cat_features
        ]

        return X_np, exposure_np, feature_cols, cat_feature_indices

    @staticmethod
    def _make_pool(
        X: np.ndarray,
        y: np.ndarray,
        cat_idxs: list[int],
    ) -> cb.Pool:
        return cb.Pool(data=X, label=y, cat_features=cat_idxs)

    def _check_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before calling predict. Call fit() first.")


# ---------------------------------------------------------------------------
# ZITReport: diagnostics and visualisation
# ---------------------------------------------------------------------------


class ZITReport:
    """
    Diagnostics and visualisation for a fitted ZITModel.

    Provides calibration plots, dispersion diagnostics, Lorenz curve,
    Vuong test for model comparison, and per-head feature importance.

    Parameters
    ----------
    model:
        A fitted ZITModel instance.
    """

    def __init__(self, model: ZITModel) -> None:
        if not model._is_fitted:
            raise ValueError("ZITReport requires a fitted ZITModel.")
        self.model = model

    def calibration_plot(
        self,
        X: pl.DataFrame,
        y: pl.Series,
        n_buckets: int = 10,
        ax=None,
    ):
        """
        Double-lift chart: predicted E[Y] vs observed mean Y, by prediction decile.

        Policies are sorted by predicted E[Y] into n_buckets equal groups.
        The plot shows predicted vs observed aggregate loss per bucket.

        Parameters
        ----------
        X:
            Feature DataFrame.
        y:
            Observed aggregate losses.
        n_buckets:
            Number of equal-count decile groups.
        ax:
            Matplotlib axes (optional).

        Returns
        -------
        matplotlib.figure.Figure
        """
        import matplotlib.pyplot as plt

        predicted = self.model.predict(X).to_numpy()
        observed = y.to_numpy()

        order = np.argsort(predicted)
        pred_sorted = predicted[order]
        obs_sorted = observed[order]

        bucket_size = len(predicted) // n_buckets
        pred_means = []
        obs_means = []

        for i in range(n_buckets):
            start = i * bucket_size
            end = (i + 1) * bucket_size if i < n_buckets - 1 else len(predicted)
            pred_means.append(np.mean(pred_sorted[start:end]))
            obs_means.append(np.mean(obs_sorted[start:end]))

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 5))
        else:
            fig = ax.get_figure()

        ax.plot(range(1, n_buckets + 1), obs_means, "o-", label="Observed", color="steelblue")
        ax.plot(range(1, n_buckets + 1), pred_means, "s--", label="Predicted E[Y]", color="tomato")
        ax.set_xlabel("Prediction decile")
        ax.set_ylabel("Mean aggregate loss")
        ax.set_title("ZIT-DGLM Calibration Plot")
        ax.legend()
        ax.grid(alpha=0.3)

        return fig

    def zero_calibration_plot(
        self,
        X: pl.DataFrame,
        y: pl.Series,
        n_buckets: int = 10,
        ax=None,
    ):
        """
        Zero calibration: predicted Pr(Y=0) vs empirical zero rate by decile.

        Parameters
        ----------
        X:
            Feature DataFrame.
        y:
            Observed aggregate losses.
        n_buckets:
            Number of buckets.
        ax:
            Matplotlib axes (optional).

        Returns
        -------
        matplotlib.figure.Figure
        """
        import matplotlib.pyplot as plt

        pred_zero = self.model.predict_proba_zero(X).to_numpy()
        observed_zero = (y.to_numpy() <= 0.0).astype(float)

        order = np.argsort(pred_zero)
        pred_sorted = pred_zero[order]
        obs_sorted = observed_zero[order]

        bucket_size = len(pred_zero) // n_buckets
        pred_means = []
        obs_means = []

        for i in range(n_buckets):
            start = i * bucket_size
            end = (i + 1) * bucket_size if i < n_buckets - 1 else len(pred_zero)
            pred_means.append(np.mean(pred_sorted[start:end]))
            obs_means.append(np.mean(obs_sorted[start:end]))

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 5))
        else:
            fig = ax.get_figure()

        ax.plot(range(1, n_buckets + 1), obs_means, "o-", label="Observed zero rate", color="steelblue")
        ax.plot(range(1, n_buckets + 1), pred_means, "s--", label="Predicted Pr(Y=0)", color="tomato")
        ax.set_xlabel("Prediction decile (by Pr(Y=0))")
        ax.set_ylabel("Zero rate / Pr(Y=0)")
        ax.set_title("ZIT-DGLM Zero Calibration Plot")
        ax.legend()
        ax.grid(alpha=0.3)

        return fig

    def dispersion_plot(
        self,
        X: pl.DataFrame,
        y: pl.Series,
        ax=None,
    ):
        """
        Dispersion diagnostic: predicted phi vs scaled unit deviance.

        Under correct specification, the scaled deviance D(y;mu)/phi should
        follow a chi-squared(1) distribution with mean 1. This plot shows
        predicted phi vs actual scaled deviance by decile.

        Parameters
        ----------
        X:
            Feature DataFrame.
        y:
            Observed aggregate losses.
        ax:
            Matplotlib axes (optional).

        Returns
        -------
        matplotlib.figure.Figure
        """
        import matplotlib.pyplot as plt

        components = self.model.predict_components(X)
        mu = components["mu"].to_numpy()
        phi = components["phi"].to_numpy()
        y_np = y.to_numpy()

        unit_dev = tweedie_unit_deviance(y_np, mu, self.model.tweedie_power)
        scaled_dev = unit_dev / np.maximum(phi, 1e-10)

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 5))
        else:
            fig = ax.get_figure()

        order = np.argsort(phi)
        phi_sorted = phi[order]
        dev_sorted = scaled_dev[order]

        n_buckets = min(10, len(phi) // 5)
        bucket_size = len(phi) // n_buckets
        phi_means = []
        dev_means = []

        for i in range(n_buckets):
            start = i * bucket_size
            end = (i + 1) * bucket_size if i < n_buckets - 1 else len(phi)
            phi_means.append(np.mean(phi_sorted[start:end]))
            dev_means.append(np.mean(dev_sorted[start:end]))

        ax.plot(phi_means, dev_means, "o-", color="steelblue", label="Mean scaled deviance")
        ax.axhline(1.0, color="tomato", linestyle="--", label="Expected = 1")
        ax.set_xlabel("Predicted dispersion phi")
        ax.set_ylabel("Scaled unit deviance D/phi")
        ax.set_title("ZIT-DGLM Dispersion Diagnostic")
        ax.legend()
        ax.grid(alpha=0.3)

        return fig

    def lorenz_curve(
        self,
        X: pl.DataFrame,
        y: pl.Series,
        ax=None,
    ):
        """
        Lorenz curve and Gini index for model discrimination.

        Policies are sorted by predicted E[Y] (ascending). The Lorenz curve
        plots the cumulative share of predicted risk vs cumulative share of
        observed claims. The Gini coefficient summarises separation quality.

        Parameters
        ----------
        X:
            Feature DataFrame.
        y:
            Observed aggregate losses.
        ax:
            Matplotlib axes (optional).

        Returns
        -------
        tuple[matplotlib.figure.Figure, float]
            Figure and Gini coefficient.
        """
        import matplotlib.pyplot as plt

        predicted = self.model.predict(X).to_numpy()
        observed = y.to_numpy()

        order = np.argsort(predicted)
        obs_sorted = observed[order]
        pred_sorted = predicted[order]

        cum_obs = np.cumsum(obs_sorted) / max(np.sum(obs_sorted), 1e-10)
        cum_pred = np.cumsum(pred_sorted) / max(np.sum(pred_sorted), 1e-10)
        n = len(predicted)
        cum_pop = np.arange(1, n + 1) / n

        # Gini from area between curve and diagonal
        gini = 1.0 - 2.0 * float(np.trapz(cum_obs, cum_pop))

        if ax is None:
            fig, ax = plt.subplots(figsize=(7, 7))
        else:
            fig = ax.get_figure()

        ax.plot(cum_pop, cum_obs, color="steelblue", label=f"ZIT (Gini={gini:.3f})")
        ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Random")
        ax.set_xlabel("Cumulative share of policies")
        ax.set_ylabel("Cumulative share of observed losses")
        ax.set_title("Lorenz Curve")
        ax.legend()
        ax.grid(alpha=0.3)

        return fig, gini

    def vuong_test(
        self,
        model_1: ZITModel,
        model_2: ZITModel,
        X: pl.DataFrame,
        y: pl.Series,
    ) -> VuongResult:
        """
        Vuong (1989) non-nested model comparison test.

        Tests H0: the two models are equally close to the true DGP.
        Positive statistic favours model_1; negative favours model_2.

        Uses the log-likelihood ratio per observation:
            LR_i = log p_1(y_i | x_i) - log p_2(y_i | x_i)

        The test statistic is:
            V = sqrt(n) * mean(LR) / std(LR)

        Under H0, V ~ N(0,1). |V| > 1.96 rejects at 5%.

        Parameters
        ----------
        model_1:
            First fitted ZITModel.
        model_2:
            Second fitted ZITModel.
        X:
            Feature DataFrame.
        y:
            Observed aggregate losses.

        Returns
        -------
        VuongResult
        """
        from scipy import stats

        data = self.model._prepare_data(X, y)
        y_np = data.y_np
        w_np = data.exposure_np

        def _compute_ll(m: ZITModel) -> np.ndarray:
            X_np_m, _, _, cat_idxs = m._extract_features(X)
            mu, phi, q = m._predict_components_np(X_np_m, cat_idxs)
            return zit_log_likelihood(y_np, mu, phi, q, m.tweedie_power, w_np)

        ll1 = _compute_ll(model_1)
        ll2 = _compute_ll(model_2)
        lr = ll1 - ll2

        n = len(lr)
        v_stat = float(np.sqrt(n) * np.mean(lr) / max(np.std(lr, ddof=1), 1e-10))
        p_value = float(2.0 * (1.0 - stats.norm.cdf(abs(v_stat))))

        if v_stat > 1.96:
            preferred = "model_1"
        elif v_stat < -1.96:
            preferred = "model_2"
        else:
            preferred = "indeterminate"

        return VuongResult(
            statistic=v_stat,
            p_value=p_value,
            preferred_model=preferred,
            n_observations=n,
            log_likelihood_ratios=lr,
        )

    def feature_importance(self, component: str = "mean") -> pl.DataFrame:
        """
        Feature importance for the specified ZIT-DGLM head.

        Returns CatBoost's built-in feature importance (PredictionValuesChange).

        Parameters
        ----------
        component:
            One of 'mean', 'dispersion', 'zero'.

        Returns
        -------
        polars.DataFrame
            Columns: feature, importance. Sorted by importance descending.
        """
        booster = self.model.get_booster(component)
        if booster is None:
            if component == "zero" and self.model.link_scenario == "linked":
                raise ValueError(
                    "No separate zero-inflation model in linked scenario. "
                    "q is derived from the mean model."
                )
            raise ValueError(f"No booster fitted for component {component!r}")

        feature_names = self.model._feature_names
        importances = booster.get_feature_importance()

        return pl.DataFrame(
            {
                "feature": feature_names,
                "importance": importances,
            }
        ).sort("importance", descending=True)
