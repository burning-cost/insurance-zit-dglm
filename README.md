# insurance-zit-dglm

Zero-Inflated Tweedie Double GLM with CatBoost gradient boosting, built for UK insurance pricing.

## The problem

Standard compound Poisson Tweedie models handle the probability of zero claims through the Poisson component: `Pr(Y=0) = exp(-lambda)`. This is fine when all policyholders are genuinely exposed — they just happen to have no claims in the period.

UK personal lines has a structural problem that breaks this assumption: **strategic non-claimers**. Under the No Claims Discount system, policyholders without protected NCD face up to 65% premium uplift for a single claim. For repairs costing less than the excess plus NCD impact, rational policyholders never claim. These are not Poisson draws — they are a distinct regime with genuinely zero claim probability.

The same phenomenon appears in home accidental damage (below-excess events unreported), motor fleet (seasonal/off-road vehicles), and subsidence (geological zero-risk properties).

The standard Tweedie conflates these two regimes. A hurdle (two-part) model goes too far in the other direction — it treats all zeros as structural, losing the compound Poisson structure that gives you the correct aggregate loss distribution.

Zero-Inflated Tweedie is the middle ground: some zeros are structural, the rest are Poisson draws. This library implements it.

## The model

The ZIT distribution mixes a point mass at zero with a standard compound Poisson-Gamma Tweedie:

```
f(y) = q * I(y=0) + (1-q) * Tweedie(mu, phi, p)
```

Parameters:
- `q` in [0,1]: structural zero probability (learned per-policy from features)
- `mu` > 0: Tweedie mean conditional on non-structural-zero
- `phi` > 0: Tweedie dispersion (DGLM extension: this is covariate-driven, not fixed)
- `p` in (1,2): Tweedie power parameter

Expected aggregate loss: `E[Y] = (1-q) * mu`

Three separate CatBoost models are fitted inside an EM loop (Gu arXiv:2405.14990):
- **Mean head** (log link): ZIT Tweedie custom loss with exposure-weighted gradients
- **Dispersion head** (log link): Smyth-Jorgensen gamma pseudo-likelihood on unit deviances
- **Zero-inflation head** (logit link): EM-weighted logistic regression with soft labels

The EM algorithm handles the unobserved indicator `z_i` (whether observation `i` is a structural zero). E-step computes posterior `Pi_i = P(z_i=1 | y_i=0, x_i)`. M-step updates all three models with EM weights that down-weight observations likely to be structural zeros.

Why the DGLM matters: the E-step depends on `mu^(2-p) / (phi*(2-p))`. If `phi` is misspecified as constant, the posterior weights `Pi_i` are wrong, contaminating all three models through the EM loop. Modelling `phi` as covariate-driven is not optional.

## Installation

```bash
pip install insurance-zit-dglm
```

## Quick start

```python
import polars as pl
from insurance_zit_dglm import ZITModel, ZITReport, check_balance

# Fit
model = ZITModel(
    tweedie_power=1.5,
    n_estimators=200,
    em_iterations=20,
    exposure_col="exposure_years",
)
model.fit(X_train, y_train)

# Predict aggregate expected loss E[Y] = (1-q)*mu
e_y = model.predict(X_test)

# All components
components = model.predict_components(X_test)
# components: mu, phi, q, E_Y

# Full P(Y=0) = q + (1-q)*exp(-mu^(2-p)/(phi*(2-p)))
prob_zero = model.predict_proba_zero(X_test)

# Balance check
result = check_balance(model, X_test, y_test, groups=age_band_series)
print(result.ratio)  # sum(E[Y]) / sum(y)
print(result.is_balanced)
```

## Diagnostic reports

```python
report = ZITReport(model)

# Calibration: observed vs predicted E[Y] by decile
report.calibration_plot(X_test, y_test)

# Zero calibration: Pr(Y=0) predicted vs empirical
report.zero_calibration_plot(X_test, y_test)

# Dispersion diagnostic: D(y;mu)/phi should be ~1
report.dispersion_plot(X_test, y_test)

# Lorenz curve and Gini
fig, gini = report.lorenz_curve(X_test, y_test)

# Vuong test: is ZIT significantly better than standard Tweedie?
from insurance_zit_dglm import ZITModel
tweedie_only = ZITModel(tweedie_power=1.5)
tweedie_only.fit(X_train, y_train)
result = report.vuong_test(model, tweedie_only, X_test, y_test)
print(result.preferred_model)  # 'model_1' | 'model_2' | 'indeterminate'

# Feature importance per head
report.feature_importance("mean")      # mu model
report.feature_importance("dispersion") # phi model
report.feature_importance("zero")       # pi model
```

## Link scenarios

**Independent** (default, recommended): three separate trees for mu, phi, and pi. The most general form — no structural relationship assumed between q and mu.

```python
model = ZITModel(link_scenario="independent")
```

**Linked**: single tree for mu; q derived as `q = 1/(1 + mu^gamma)`. This enforces the economic intuition that higher-risk policies are less likely to be structural zeros (So & Valdez arXiv:2406.16206 Scenario 2). If `gamma=None`, it is estimated by grid search.

```python
model = ZITModel(link_scenario="linked", gamma=1.0)
```

## Power parameter

The Tweedie power `p` is not gradient-boosted — it is estimated separately by profile likelihood. Use `estimate_power()` to select it before fitting:

```python
from insurance_zit_dglm import estimate_power

# Quick grid search with initial mu estimates
p_hat = estimate_power(y_train.to_numpy(), mu_initial, p_grid=[1.2, 1.3, 1.4, 1.5, 1.6, 1.7])
model = ZITModel(tweedie_power=p_hat)
```

## Autocalibration

Gradient boosting minimising ZIT deviance does not automatically satisfy the balance property `sum(E[Y_i]) = sum(y_i)`. For FCA Consumer Duty compliance, check this explicitly:

```python
result = check_balance(model, X_val, y_val, tolerance=0.02)
if not result.is_balanced:
    from insurance_zit_dglm import recalibrate
    recal_model = recalibrate(model, X_val, y_val)
    # recal_model applies a multiplicative intercept correction
```

## Mathematical foundation

- Gu (arXiv:2405.14990): ZIT with dispersion modelling and generalised EM algorithm
- So & Valdez (arXiv:2406.16206 / NAAJ Vol 29(4):887-904, 2025): ZIT boosted trees, CatBoost implementation, Vuong test
- Delong & Wuthrich (arXiv:2103.03635): balance property and autocalibration

## UK peril guidance

| Peril | ZIT recommended? | Reason |
|-------|-----------------|--------|
| Motor AD (non-protected NCD) | Yes | NCD behavioural zeros |
| Home accidental damage | Yes | Sub-excess strategic non-claiming |
| Subsidence | Yes | Geological regime effect |
| Commercial fleet | Yes | Seasonal/off-road structural zeros |
| Comprehensive motor (protected NCD) | Marginal | Standard Tweedie often sufficient |
| Home escape of water | No | Genuine compound Poisson |
| Motor windscreen | No | Low excess, few strategic zeros |
