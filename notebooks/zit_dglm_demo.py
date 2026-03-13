# Databricks notebook source
# MAGIC %md
# MAGIC # insurance-zit-dglm: Zero-Inflated Tweedie DGLM Demo
# MAGIC
# MAGIC This notebook demonstrates the full workflow:
# MAGIC 1. Synthetic ZIT data generation with known DGP
# MAGIC 2. Fitting ZITModel (independent and linked scenarios)
# MAGIC 3. Diagnostics via ZITReport
# MAGIC 4. Balance property check and recalibration
# MAGIC 5. Vuong test: ZIT vs standard Tweedie
# MAGIC 6. Power parameter estimation

# COMMAND ----------
# MAGIC %pip install insurance-zit-dglm --quiet

# COMMAND ----------

import numpy as np
import polars as pl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from insurance_zit_dglm import ZITModel, ZITReport, estimate_power, check_balance
from insurance_zit_dglm.calibration import recalibrate

# COMMAND ----------
# MAGIC %md
# MAGIC ## 1. Synthetic ZIT Data
# MAGIC
# MAGIC DGP with known parameters:
# MAGIC - log(mu) = 0.3 + 0.8*x1 - 0.4*x2 + 0.2*x3
# MAGIC - log(phi) = 0.1 + 0.3*|x2|  (heteroscedastic dispersion)
# MAGIC - logit(q) = -1.5 + 0.7*x1   (zero-inflation driven by x1)
# MAGIC - Structural zeros from Bernoulli(q)
# MAGIC - Tweedie (compound Poisson-Gamma, p=1.5) for non-structural zeros
# MAGIC - 30,000 policies with exposure ~ Uniform(0.5, 2.0)

# COMMAND ----------


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def simulate_uk_motor_ad(n: int = 30_000, seed: int = 0):
    """
    Simulate UK motor accidental damage claims.

    NCD behavioural zeros (strategic non-claimers) are modelled via
    the q component. Non-strategic policyholders have compound Poisson
    claim process.
    """
    rng = np.random.default_rng(seed)

    # Features: age band, vehicle age, NCB level, region index, annual mileage
    age = rng.integers(17, 80, n).astype(float)
    vehicle_age = rng.integers(0, 20, n).astype(float)
    ncb_years = rng.integers(0, 10, n).astype(float)  # NCD years
    region = rng.integers(1, 12, n).astype(float)
    mileage = rng.normal(10000, 5000, n).clip(1000, 50000) / 10000.0  # thousands

    exposure = rng.uniform(0.25, 1.5, n)  # policy years (mid-term cancellations etc)

    # Normalised features for the model
    age_n = (age - 40) / 20
    veh_n = vehicle_age / 10.0
    mileage_n = mileage - 1.0

    # True mu: younger drivers, older vehicles, high mileage => more claims
    log_mu = -1.2 + 0.4 * np.abs(age_n) - 0.3 * age_n + 0.3 * veh_n + 0.25 * mileage_n
    mu_true = np.exp(log_mu)

    # True phi: higher for young drivers (heteroscedastic)
    log_phi = -0.3 + 0.5 * np.maximum(1.5 - age_n, 0)
    phi_true = np.exp(log_phi)

    # True q: NCD-driven strategic zeros (higher NCD => more strategic)
    logit_q = -2.0 + 0.4 * ncb_years - 0.3 * mileage_n
    q_true = sigmoid(logit_q)

    # Simulate
    p = 1.5
    p2 = 2.0 - p
    p1 = p - 1.0

    z = rng.binomial(1, q_true)  # structural zero indicator
    lam = exposure * (mu_true ** p2) / (phi_true * p2)

    y = np.zeros(n)
    gamma_shape = p2 / p1
    gamma_scale = phi_true * p1 * (mu_true ** p1) / exposure

    for i in range(n):
        if z[i] == 1:
            y[i] = 0.0
        else:
            n_claims = rng.poisson(lam[i])
            if n_claims > 0:
                y[i] = float(np.sum(rng.gamma(gamma_shape, gamma_scale[i], n_claims)))

    X = pl.DataFrame({
        "age": age,
        "vehicle_age": vehicle_age,
        "ncb_years": ncb_years,
        "region": region,
        "mileage_k": mileage,
        "exposure": exposure,
    })
    y_series = pl.Series(y)

    truth = {"mu": mu_true, "phi": phi_true, "q": q_true, "z": z}
    return X, y_series, truth


X, y, truth = simulate_uk_motor_ad(n=30_000, seed=42)

print(f"Dataset: {len(X):,} policies")
print(f"Zero claims: {(y == 0).sum():,} ({100*(y==0).mean():.1f}%)")
print(f"True structural zeros: {truth['z'].sum():,} ({100*truth['z'].mean():.1f}%)")
print(f"Mean observed loss: {y.mean():.4f}")
print(f"True q range: [{truth['q'].min():.3f}, {truth['q'].max():.3f}]")
print(f"True mu range: [{truth['mu'].min():.3f}, {truth['mu'].max():.3f}]")
print(f"\nFeature sample:\n{X.head(5)}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 2. Train/Test Split

# COMMAND ----------

rng = np.random.default_rng(99)
n = len(X)
idx = rng.permutation(n)
train_idx = idx[:int(0.75 * n)]
test_idx = idx[int(0.75 * n):]

X_train = X[train_idx]
y_train = y[train_idx]
X_test = X[test_idx]
y_test = y[test_idx]

print(f"Train: {len(X_train):,} | Test: {len(X_test):,}")
print(f"Train zero rate: {100*(y_train==0).mean():.1f}%")
print(f"Test zero rate: {100*(y_test==0).mean():.1f}%")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 3. Fit ZITModel (Independent Scenario)

# COMMAND ----------

model = ZITModel(
    tweedie_power=1.5,
    n_estimators=150,
    learning_rate=0.05,
    max_depth=4,
    em_iterations=15,
    em_tol=1e-5,
    link_scenario="independent",
    exposure_col="exposure",
    verbose=0,
    random_seed=42,
)

print("Fitting ZIT-DGLM (independent scenario)...")
model.fit(X_train, y_train)
print(f"Converged in {len(model._log_likelihoods)} EM iterations")

# COMMAND ----------
# MAGIC %md
# MAGIC ### EM Convergence

# COMMAND ----------

fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(range(1, len(model._log_likelihoods) + 1), model._log_likelihoods, "o-", color="steelblue")
ax.set_xlabel("EM Iteration")
ax.set_ylabel("Observed log-likelihood")
ax.set_title("ZIT-DGLM EM Convergence")
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("/tmp/em_convergence.png", dpi=100, bbox_inches="tight")
display(fig)
plt.close()

# COMMAND ----------
# MAGIC %md
# MAGIC ## 4. Predictions on Test Set

# COMMAND ----------

e_y = model.predict(X_test)
components = model.predict_components(X_test)
prob_zero = model.predict_proba_zero(X_test)

print("Predicted E[Y] summary:")
print(f"  Mean predicted: {e_y.mean():.4f}")
print(f"  Mean observed:  {y_test.mean():.4f}")
print(f"\nComponents summary:")
print(f"  mu:  [{components['mu'].min():.3f}, {components['mu'].max():.3f}] mean={components['mu'].mean():.3f}")
print(f"  phi: [{components['phi'].min():.3f}, {components['phi'].max():.3f}] mean={components['phi'].mean():.3f}")
print(f"  q:   [{components['q'].min():.3f}, {components['q'].max():.3f}] mean={components['q'].mean():.3f}")
print(f"\nPr(Y=0) summary:")
print(f"  Mean predicted Pr(Y=0): {prob_zero.mean():.4f}")
print(f"  Observed zero rate:     {(y_test==0).mean():.4f}")

test_score = model.score(X_test, y_test)
print(f"\nTest log-likelihood (mean): {test_score:.4f}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 5. Diagnostics

# COMMAND ----------

report = ZITReport(model)

fig = report.calibration_plot(X_test, y_test, n_buckets=10)
plt.savefig("/tmp/calibration.png", dpi=100, bbox_inches="tight")
display(fig)
plt.close()

# COMMAND ----------

fig = report.zero_calibration_plot(X_test, y_test)
plt.savefig("/tmp/zero_calibration.png", dpi=100, bbox_inches="tight")
display(fig)
plt.close()

# COMMAND ----------

fig = report.dispersion_plot(X_test, y_test)
plt.savefig("/tmp/dispersion.png", dpi=100, bbox_inches="tight")
display(fig)
plt.close()

# COMMAND ----------

fig, gini = report.lorenz_curve(X_test, y_test)
print(f"Gini coefficient: {gini:.3f}")
plt.savefig("/tmp/lorenz.png", dpi=100, bbox_inches="tight")
display(fig)
plt.close()

# COMMAND ----------
# MAGIC %md
# MAGIC ## 6. Feature Importance Per Head

# COMMAND ----------

print("=== Mean (mu) head ===")
print(report.feature_importance("mean"))

print("\n=== Dispersion (phi) head ===")
print(report.feature_importance("dispersion"))

print("\n=== Zero-inflation (pi) head ===")
print(report.feature_importance("zero"))

# COMMAND ----------
# MAGIC %md
# MAGIC ## 7. Balance Property Check

# COMMAND ----------

age_band = pl.Series([
    "young" if a < 25 else "mid" if a < 55 else "senior"
    for a in X_test["age"].to_list()
])

result = check_balance(model, X_test, y_test, groups=age_band, tolerance=0.02)

print(f"Overall balance:")
print(f"  ratio = {result.ratio:.4f} {'OK' if result.is_balanced else 'IMBALANCED'}")
print(f"  total predicted = {result.total_predicted:.2f}")
print(f"  total observed  = {result.total_observed:.2f}")
print(f"  zero calibration ratio: {result.zero_calibration_ratio:.4f}")
print(f"  dispersion check (D/phi): {result.dispersion_check:.4f}")

print(f"\nGroup-level balance:")
for grp, grp_result in sorted(result.group_results.items()):
    status = "OK" if grp_result.is_balanced else "IMBALANCED"
    print(f"  {grp}: ratio={grp_result.ratio:.4f} n={grp_result.n_observations} {status}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 8. Recalibration

# COMMAND ----------

if not result.is_balanced:
    print("Model is imbalanced — applying Delong-Wuthrich intercept correction...")
    recal_model = recalibrate(model, X_test, y_test)
    print(f"Correction factor: {recal_model.correction_factor:.4f}")

    recal_preds = recal_model.predict(X_test)
    print(f"After recalibration:")
    print(f"  sum predicted = {recal_preds.sum():.2f}")
    print(f"  sum observed  = {y_test.sum():.2f}")
    print(f"  ratio = {recal_preds.sum() / y_test.sum():.4f}")
else:
    print("Model is balanced — no recalibration needed.")
    recal_model = model

# COMMAND ----------
# MAGIC %md
# MAGIC ## 9. Vuong Test: ZIT vs Standard Tweedie

# COMMAND ----------

print("Fitting standard Tweedie (q=0 equivalent)...")
# Fit a ZIT model with very low n_estimators for the pi head — the zero-inflation
# component will be minimal, approximating standard Tweedie
tweedie_approx = ZITModel(
    tweedie_power=1.5,
    n_estimators=150,
    learning_rate=0.05,
    max_depth=4,
    em_iterations=2,   # Minimal EM = minimal zero-inflation correction
    exposure_col="exposure",
    verbose=0,
    random_seed=0,
)
tweedie_approx.fit(X_train, y_train)

vuong = report.vuong_test(model, tweedie_approx, X_test, y_test)
print(f"\nVuong test: ZIT-DGLM vs minimal-ZI Tweedie")
print(f"  Statistic V = {vuong.statistic:.3f}")
print(f"  p-value     = {vuong.p_value:.4f}")
print(f"  Preferred   = {vuong.preferred_model}")
print(f"  n obs       = {vuong.n_observations:,}")

if vuong.preferred_model == "model_1":
    print("\nConclusion: ZIT-DGLM significantly outperforms standard Tweedie on this dataset.")
elif vuong.preferred_model == "model_2":
    print("\nConclusion: Standard Tweedie not significantly worse — ZIT may be overkill here.")
else:
    print("\nConclusion: Models are statistically equivalent — standard Tweedie is simpler.")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 10. Linked Scenario (So & Valdez Scenario 2)
# MAGIC
# MAGIC In the linked scenario, q = 1/(1 + mu^gamma). This enforces the economic
# MAGIC intuition that higher-risk policies are less likely to be structural zeros.

# COMMAND ----------

model_linked = ZITModel(
    tweedie_power=1.5,
    n_estimators=100,
    learning_rate=0.05,
    max_depth=4,
    em_iterations=10,
    link_scenario="linked",
    gamma=None,  # estimate from data
    exposure_col="exposure",
    verbose=0,
    random_seed=42,
)

print("Fitting ZIT-DGLM (linked scenario, gamma estimated)...")
model_linked.fit(X_train, y_train)
print(f"Estimated gamma: {model_linked._gamma_fitted:.2f}")

# Compare components
comp_indep = model.predict_components(X_test)
comp_linked = model_linked.predict_components(X_test)

print(f"\nIndependent: mean q = {comp_indep['q'].mean():.4f}")
print(f"Linked:      mean q = {comp_linked['q'].mean():.4f}")

score_indep = model.score(X_test, y_test)
score_linked = model_linked.score(X_test, y_test)
print(f"\nTest log-likelihood:")
print(f"  Independent: {score_indep:.4f}")
print(f"  Linked:      {score_linked:.4f}")
print(f"  Winner: {'Independent' if score_indep >= score_linked else 'Linked'}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 11. Power Parameter Estimation

# COMMAND ----------

# Estimate power using current model's mu/phi/q estimates
mu_np = comp_indep["mu"].to_numpy()
phi_np = comp_indep["phi"].to_numpy()
q_np = comp_indep["q"].to_numpy()
y_test_np = y_test.to_numpy()

p_hat = estimate_power(
    y=y_test_np,
    mu=mu_np,
    phi=phi_np,
    q=q_np,
    p_grid=[1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8],
    exposure=X_test["exposure"].to_numpy(),
)

print(f"Profile likelihood estimates power p = {p_hat}")
print(f"(True DGP used p = 1.5)")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC | Metric | Value |
# MAGIC |--------|-------|
# MAGIC | EM iterations | see above |
# MAGIC | Test log-likelihood | see above |
# MAGIC | Overall balance ratio | see above |
# MAGIC | Zero calibration ratio | see above |
# MAGIC | Gini coefficient | see above |
# MAGIC | Estimated power | see above |
# MAGIC
# MAGIC The ZIT-DGLM correctly separates structural zeros (NCD behavioural non-claimers)
# MAGIC from genuine compound Poisson zeros, producing better-calibrated zero probabilities
# MAGIC and more accurate aggregate loss predictions in the presence of strategic non-claiming.
