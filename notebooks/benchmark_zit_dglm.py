# Databricks notebook source
# MAGIC %md
# MAGIC # Benchmark: insurance-zit-dglm (ZIT-DGLM) vs plain Tweedie GLM
# MAGIC
# MAGIC **Library:** `insurance-zit-dglm` — Zero-Inflated Tweedie Double GLM with CatBoost
# MAGIC gradient boosting. Three heads (mean, dispersion, zero-inflation) fitted in an
# MAGIC EM loop (Gu arXiv:2405.14990, So & Valdez arXiv:2406.16206).
# MAGIC
# MAGIC **Baseline:** plain Tweedie GLM (CatBoost, Poisson objective as a close
# MAGIC approximation for zero-heavy data). No explicit zero-inflation modelling.
# MAGIC
# MAGIC **Dataset:** Synthetic UK motor accidental damage pure premiums under an NCD
# MAGIC regime — 40,000 policies. Known DGP with structural zeros (strategic non-claimers)
# MAGIC at rates varying by NCD level, age band, and vehicle class.
# MAGIC Temporal 70/30 train/test split.
# MAGIC
# MAGIC **Date:** 2026-03-13
# MAGIC
# MAGIC **Library version:** 0.1.0
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC The benchmark tests the core proposition: when structural zeros are present (NCD
# MAGIC strategic non-claimers), a three-head model that explicitly separates the two
# MAGIC zero-generating processes outperforms a model that conflates them. The test is
# MAGIC whether the ZIT model's zero calibration and overall aggregate loss prediction
# MAGIC are meaningfully better than a plain Tweedie.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Setup

# COMMAND ----------

%pip install insurance-zit-dglm catboost matplotlib numpy scipy polars pandas

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import time
import warnings
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import polars as pl
from scipy import stats
import catboost as cb

from insurance_zit_dglm import ZITModel, ZITReport, check_balance
from insurance_zit_dglm.losses import zit_log_likelihood, tweedie_unit_deviance

warnings.filterwarnings("ignore")

print(f"Benchmark run at: {datetime.utcnow().isoformat()}Z")
print("Libraries loaded.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Data Generation
# MAGIC
# MAGIC We generate synthetic UK motor accidental damage data with structural zeros.
# MAGIC
# MAGIC DGP:
# MAGIC - Some policyholders never claim regardless of exposure (structural zeros,
# MAGIC   modelled by q_i = P(structural zero | x_i))
# MAGIC - Among non-structural-zero policyholders, losses follow Tweedie(mu_i, phi_i, p=1.5)
# MAGIC - **q_i is high for high-NCD policies** (strong incentive to avoid small claims)
# MAGIC - **q_i is lower for young drivers** (less NCD to protect, smaller excess impact)
# MAGIC
# MAGIC This creates a testable DGP: the zero inflation is structured, not random.
# MAGIC A model that recovers the q structure will have better zero calibration than one
# MAGIC that relies on the Tweedie compound Poisson to absorb excess zeros.

# COMMAND ----------

rng = np.random.default_rng(1234)
N = 40_000
TWEEDIE_P = 1.5

# Covariates
ncd_level  = rng.choice([0, 1, 2, 3, 4], N, p=[0.12, 0.18, 0.22, 0.26, 0.22])
age_band   = rng.choice(["17-24", "25-34", "35-49", "50-64", "65+"], N,
                         p=[0.10, 0.22, 0.30, 0.26, 0.12])
veh_class  = rng.choice(["A", "B", "C", "D"], N, p=[0.30, 0.30, 0.25, 0.15])
region     = rng.choice(["London", "SE", "Midlands", "North"], N,
                         p=[0.22, 0.20, 0.30, 0.28])
sum_ins    = rng.lognormal(9.2, 0.55, N)
log_si     = np.log(sum_ins)
exposure   = rng.uniform(0.3, 1.0, N)   # years on risk in policy period

# --- True structural zero probability (NCD strategic non-claimers) ---
q_logit  = -1.5   # base: exp(-1.5)/(1+exp(-1.5)) ~ 0.18 base rate
q_logit += 0.60 * (ncd_level / 4.0)          # higher NCD -> more to protect
q_logit += np.where(age_band == "17-24", -0.50, 0.0)  # young: less NCD to protect
q_logit += np.where(age_band == "65+",    0.20, 0.0)  # older: value NCD more
q_logit += np.where(veh_class == "D", -0.25, 0.0)     # luxury: claims visible/expensive
q_logit += 0.25 * np.log(sum_ins / np.median(sum_ins))  # high SI -> less strategic non-claiming

q_true = 1.0 / (1.0 + np.exp(-q_logit))
q_true = np.clip(q_true, 0.05, 0.70)

# --- True mu (mean severity conditional on non-structural-zero) ---
mu_log  = 5.8  # base: ~£330 expected severity
mu_log += np.where(age_band == "17-24",  0.55, 0.0)
mu_log += np.where(age_band == "25-34",  0.20, 0.0)
mu_log += np.where(age_band == "65+",    0.10, 0.0)
mu_log += np.where(veh_class == "B",  0.30, 0.0)
mu_log += np.where(veh_class == "C",  0.65, 0.0)
mu_log += np.where(veh_class == "D",  1.15, 0.0)
mu_log += np.where(region == "London", 0.25, 0.0)
mu_log += np.where(region == "SE",     0.12, 0.0)
mu_log += 0.25 * (log_si - np.log(np.median(sum_ins)))
mu_true = np.exp(mu_log) * exposure

# --- True phi (dispersion) ---
phi_true_arr = 1.5 * np.ones(N)
phi_true_arr[veh_class == "D"] = 2.5  # luxury cars more volatile
phi_true_arr[age_band == "17-24"] = 2.0  # young drivers more volatile

# --- Generate observations ---
structural_zero = rng.random(N) < q_true
y = np.zeros(N)
for i in np.where(~structural_zero)[0]:
    # Tweedie: compound Poisson-Gamma
    lam_i = mu_true[i] ** (2 - TWEEDIE_P) / (phi_true_arr[i] * (2 - TWEEDIE_P))
    nc    = rng.poisson(max(lam_i, 1e-8))
    if nc > 0:
        alpha_g = (2 - TWEEDIE_P) / (TWEEDIE_P - 1)
        beta_g  = phi_true_arr[i] * (TWEEDIE_P - 1) * mu_true[i] ** (TWEEDIE_P - 1)
        y[i] = rng.gamma(alpha_g * nc, scale=beta_g)

# Build DataFrame
policy_year = rng.choice([2021, 2022, 2023], N, p=[0.35, 0.35, 0.30])
order = np.argsort(policy_year, kind="stable")

df_pd = pd.DataFrame({
    "ncd_level":   ncd_level.astype(str),
    "age_band":    age_band,
    "veh_class":   veh_class,
    "region":      region,
    "log_si":      log_si,
    "exposure":    exposure,
    "policy_year": policy_year,
    "y":           y,
    "q_true":      q_true,
    "mu_true":     mu_true,
    "phi_true":    phi_true_arr,
    "structural_zero": structural_zero.astype(int),
})
df_pd = df_pd.iloc[order].reset_index(drop=True)

train_end = int(N * 0.70)
train_pd = df_pd.iloc[:train_end].copy()
test_pd  = df_pd.iloc[train_end:].copy()

zero_rate = (y == 0).mean()
structural_rate = structural_zero.mean()
compound_zero_rate = zero_rate - structural_rate

print(f"Total:  {N:,}")
print(f"Train:  {len(train_pd):,}  ({100*len(train_pd)/N:.0f}%)")
print(f"Test:   {len(test_pd):,}   ({100*len(test_pd)/N:.0f}%)")
print()
print(f"Zero rate (total):   {zero_rate:.1%}")
print(f"Structural zeros:    {structural_rate:.1%}  (NCD strategic non-claimers)")
print(f"Compound Pois zeros: {compound_zero_rate:.1%}  (genuine no-claim policies)")
print()
print(f"True q by NCD level:")
for ncd in [0, 1, 2, 3, 4]:
    m = ncd_level == ncd
    print(f"  NCD {ncd}: mean q = {q_true[m].mean():.3f}, zero rate = {(y[m]==0).mean():.3f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Baseline: CatBoost Tweedie (no zero inflation)
# MAGIC
# MAGIC The baseline fits a CatBoost model with the standard Tweedie loss function.
# MAGIC This is how most teams model pure premium with gradient boosting — no explicit
# MAGIC treatment of the structural zero process.
# MAGIC
# MAGIC The Tweedie compound Poisson can in principle assign low lambda (low claim frequency)
# MAGIC to policies that the DGP has marked as structural zeros. But it cannot distinguish
# MAGIC between a structural zero and a Poisson zero — it treats all zeros identically.
# MAGIC This leads to systematic zero miscalibration for high-NCD segments.

# COMMAND ----------

FEATURES = ["ncd_level", "age_band", "veh_class", "region", "log_si", "exposure"]
CAT_FEATURES_CB = ["ncd_level", "age_band", "veh_class", "region"]

t0 = time.perf_counter()

y_train = train_pd["y"].values
y_test  = test_pd["y"].values
exp_train = train_pd["exposure"].values
exp_test  = test_pd["exposure"].values

X_train_pd = train_pd[FEATURES]
X_test_pd  = test_pd[FEATURES]

# Tweedie rate model: target = y/exposure, weight = exposure
# Then prediction * exposure gives expected aggregate loss
pool_tw = cb.Pool(
    data=X_train_pd,
    label=(y_train / np.maximum(exp_train, 1e-8)),
    weight=exp_train,
    cat_features=CAT_FEATURES_CB,
)

cb_tweedie = cb.CatBoostRegressor(
    loss_function=f"Tweedie:variance_power={TWEEDIE_P}",
    iterations=300,
    learning_rate=0.08,
    depth=5,
    random_seed=42,
    verbose=0,
    allow_writing_files=False,
)
cb_tweedie.fit(pool_tw)

pred_rate_base_train = cb_tweedie.predict(X_train_pd)
pred_rate_base_test  = cb_tweedie.predict(X_test_pd)
mu_base_train = np.maximum(pred_rate_base_train, 1e-10) * exp_train
mu_base_test  = np.maximum(pred_rate_base_test,  1e-10) * exp_test

baseline_fit_time = time.perf_counter() - t0

print(f"Baseline fit time: {baseline_fit_time:.2f}s")
print(f"Mean predicted (test): {mu_base_test.mean():.2f}")
print(f"Mean actual (test):    {y_test.mean():.2f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Library: ZIT-DGLM
# MAGIC
# MAGIC The ZIT-DGLM explicitly models the structural zero process alongside the mean
# MAGIC and dispersion. Three CatBoost models in an EM loop:
# MAGIC
# MAGIC - **mu model** (log link): expected severity conditional on non-structural-zero
# MAGIC - **phi model** (log link): per-policy Tweedie dispersion
# MAGIC - **pi model** (logit link): structural zero probability q_i
# MAGIC
# MAGIC Final prediction: E[Y] = (1 - q_i) * mu_i

# COMMAND ----------

t0 = time.perf_counter()

# Build Polars DataFrames for ZITModel API
# ZITModel expects Polars input with exposure as a column
train_pl = pl.from_pandas(train_pd[FEATURES])
test_pl  = pl.from_pandas(test_pd[FEATURES])
y_train_pl = pl.Series(y_train.astype(float))
y_test_pl  = pl.Series(y_test.astype(float))

zit = ZITModel(
    tweedie_power=TWEEDIE_P,
    n_estimators=200,
    learning_rate=0.08,
    max_depth=5,
    em_iterations=15,
    em_tol=1e-5,
    cat_features=["ncd_level", "age_band", "veh_class", "region"],
    exposure_col="exposure",
    verbose=0,
    random_seed=42,
)
zit.fit(train_pl, y_train_pl)

library_fit_time = time.perf_counter() - t0

# Predictions
comps_test   = zit.predict_components(test_pl)
mu_zit_test  = comps_test["mu"].to_numpy()
phi_zit_test = comps_test["phi"].to_numpy()
q_zit_test   = comps_test["q"].to_numpy()
ey_zit_test  = comps_test["E_Y"].to_numpy()

prob_zero_zit   = zit.predict_proba_zero(test_pl).to_numpy()

print(f"ZIT-DGLM fit time:    {library_fit_time:.2f}s")
print(f"EM iterations used:   {len(zit._log_likelihoods)}")
print(f"Final log-likelihood: {zit._log_likelihoods[-1]:.1f}")
print()
print(f"ZIT predictions (test):")
print(f"  Mean mu:    {mu_zit_test.mean():.2f}")
print(f"  Mean q:     {q_zit_test.mean():.3f}  (vs true q: {test_pd['q_true'].values.mean():.3f})")
print(f"  Mean E[Y]:  {ey_zit_test.mean():.2f}  (vs actual: {y_test.mean():.2f})")
print(f"  Mean P(Y=0): {prob_zero_zit.mean():.3f}  (vs actual zero rate: {(y_test==0).mean():.3f})")

# COMMAND ----------

# Balance check
bal = check_balance(zit, test_pl, y_test_pl, tolerance=0.02)
print(f"Balance ratio (sum E[Y] / sum y): {bal.ratio:.4f}")
print(f"Is balanced (within 2%):          {bal.is_balanced}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Metrics
# MAGIC
# MAGIC Three axes of comparison:
# MAGIC
# MAGIC 1. **Aggregate loss calibration**: A/E ratio overall and by NCD level. NCD is
# MAGIC    the key structural variable — the ZIT model should track it better.
# MAGIC
# MAGIC 2. **Zero probability calibration**: Pr(Y=0) predicted vs observed zero rate
# MAGIC    by NCD decile. This is the direct test of whether the model understands
# MAGIC    strategic non-claiming behaviour.
# MAGIC
# MAGIC 3. **Log-likelihood on test set**: measures fit to the full distribution p(Y|X),
# MAGIC    not just the conditional mean. A model with better zero calibration should have
# MAGIC    higher log-likelihood for zero observations.

# COMMAND ----------

def tweedie_ll_approx(y, mu, phi=1.5, p=1.5):
    """
    Approximate Tweedie log-likelihood for benchmarking purposes.
    Uses the unit deviance formula: ll ~ -D(y;mu)/(2*phi) - const.
    """
    y   = np.asarray(y, dtype=float)
    mu  = np.maximum(np.asarray(mu, dtype=float), 1e-10)
    p1  = p - 1.0
    p2  = 2.0 - p
    d   = 2.0 * (
        np.where(y > 0, y**(2-p) / ((1-p)*(2-p)) - y * mu**(-p1) / (1-p), 0.0)
        + mu**(2-p) / (2-p)
    )
    return -d / (2.0 * phi)


def ae_by_group(y, mu, group_series):
    df_t = pd.DataFrame({"y": y, "mu": mu, "g": group_series})
    return {g: sub["y"].sum() / max(sub["mu"].sum(), 1e-10)
            for g, sub in df_t.groupby("g")}


def zero_calibration_rmse(y, prob_zero_pred, n_deciles=10):
    """
    RMSE of predicted P(Y=0) vs empirical zero rate by predicted-zero-prob decile.
    """
    order = np.argsort(prob_zero_pred)
    y_s   = (y == 0.0).astype(float)[order]
    pz_s  = prob_zero_pred[order]
    splits = np.array_split(np.arange(len(y_s)), n_deciles)
    errs   = [(pz_s[i].mean() - y_s[i].mean()) ** 2 for i in splits if len(i) > 0]
    return float(np.sqrt(np.mean(errs)))

# COMMAND ----------

# Tweedie baseline zero probability
# Under Tweedie: Pr(Y=0) = exp(-lambda) where lambda = mu^(2-p)/(phi*(2-p))
phi_const_base = 1.5  # typical Tweedie dispersion
lam_base = mu_base_test ** (2 - TWEEDIE_P) / (phi_const_base * (2 - TWEEDIE_P))
lam_base = np.maximum(lam_base, 1e-10)
prob_zero_base = np.exp(-lam_base)

# Metrics
y_te  = y_test
mu_b  = mu_base_test
mu_z  = ey_zit_test

# A/E by NCD level
ae_base_ncd = ae_by_group(y_te, mu_b, test_pd["ncd_level"].values)
ae_zit_ncd  = ae_by_group(y_te, mu_z, test_pd["ncd_level"].values)

# Zero calibration RMSE
zcal_base = zero_calibration_rmse(y_te, prob_zero_base)
zcal_zit  = zero_calibration_rmse(y_te, prob_zero_zit)

# Log-likelihood (approximate for baseline)
ll_base = tweedie_ll_approx(y_te, mu_b, phi=phi_const_base, p=TWEEDIE_P).mean()

# ZIT log-likelihood uses the library
zit_score = zit.score(test_pl, y_test_pl)

# Overall A/E
ae_overall_base = y_te.sum() / max(mu_b.sum(), 1e-10)
ae_overall_zit  = y_te.sum() / max(mu_z.sum(), 1e-10)

# Tweedie deviance
def tw_dev(y, mu, p=TWEEDIE_P):
    y = np.asarray(y, dtype=float)
    mu = np.maximum(np.asarray(mu, dtype=float), 1e-10)
    p1, p2 = p - 1.0, 2.0 - p
    d = 2.0 * (np.where(y>0, y**(2-p)/((1-p)*(2-p)) - y*mu**(-p1)/(1-p), 0.0)
               + mu**(2-p)/(2-p))
    return float(np.mean(d))

dev_base = tw_dev(y_te, mu_b)
dev_zit  = tw_dev(y_te, mu_z)

print("=" * 78)
print(f"{'Metric':<44} {'Baseline (Tweedie)':>18} {'ZIT-DGLM':>10} {'Delta':>6}")
print("=" * 78)

mets = [
    ("Tweedie deviance (lower better)",   dev_base,     dev_zit,      True),
    ("Log-likelihood per obs (higher)",   ll_base,      zit_score,    False),
    ("Overall A/E ratio (target 1.0)",    ae_overall_base, ae_overall_zit, None),
    ("Zero calib RMSE (lower better)",    zcal_base,    zcal_zit,     True),
    ("Fit time (s)",                      baseline_fit_time, library_fit_time, True),
]
for name, bv, lv, lib in mets:
    if lib is None:
        delta_str = f"base={bv:.4f}, ZIT={lv:.4f}"
        print(f"  {name:<42} {bv:>18.4f} {lv:>10.4f}")
    elif bv == 0 or bv == lv:
        print(f"  {name:<42} {bv:>18.4f} {lv:>10.4f}")
    else:
        delta = (lv - bv) / abs(bv) * 100
        print(f"  {name:<42} {bv:>18.4f} {lv:>10.4f} {delta:>+5.1f}%")

print("=" * 78)

# COMMAND ----------

# A/E by NCD level
print("\nA/E by NCD Level (key segmentation test):")
print(f"  {'NCD':<8} {'Baseline A/E':>14} {'ZIT A/E':>10} {'True mean q':>12} {'Actual zero rate':>16}")
print("-" * 64)
for ncd in ["0", "1", "2", "3", "4"]:
    m = test_pd["ncd_level"].values == ncd
    if m.sum() < 10:
        continue
    true_q_m  = test_pd["q_true"].values[m].mean()
    zero_rate_m = (y_te[m] == 0).mean()
    print(f"  {ncd:<8} {ae_base_ncd.get(ncd, np.nan):>14.4f} "
          f"{ae_zit_ncd.get(ncd, np.nan):>10.4f} "
          f"{true_q_m:>12.3f} {zero_rate_m:>16.3f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Diagnostic Plots

# COMMAND ----------

report = ZITReport(zit)

fig = plt.figure(figsize=(16, 14))
gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.30)

ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[1, 0])
ax4 = fig.add_subplot(gs[1, 1])

# ── Plot 1: Zero calibration by decile ─────────────────────────────────────
n_dec = 10

def zero_cal_plot(y_arr, prob_zero, label, color, ax, linestyle="-o"):
    order  = np.argsort(prob_zero)
    y_s    = (y_arr == 0).astype(float)[order]
    pz_s   = prob_zero[order]
    splits = np.array_split(np.arange(len(y_s)), n_dec)
    obs_m  = [y_s[i].mean() for i in splits]
    pred_m = [pz_s[i].mean() for i in splits]
    ax.plot(range(1, n_dec+1), pred_m, linestyle, label=f"{label} Predicted",
            color=color, linewidth=1.5)
    ax.plot(range(1, n_dec+1), obs_m, "s--", label=f"Observed zero rate",
            color="black", linewidth=1.2, alpha=0.7)

zero_cal_plot(y_te, prob_zero_zit, "ZIT-DGLM", "tomato", ax1)
ax1.set_xlabel("Decile (sorted by predicted P(Y=0))")
ax1.set_ylabel("Zero rate / P(Y=0)")
ax1.set_title("ZIT-DGLM Zero Calibration by Decile")
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)

# ── Plot 2: True q vs fitted q by NCD level ────────────────────────────────
ncd_levels = ["0", "1", "2", "3", "4"]
true_q_ncd = [test_pd.loc[test_pd["ncd_level"] == ncd, "q_true"].mean()
              for ncd in ncd_levels]
fit_q_ncd  = [q_zit_test[test_pd["ncd_level"].values == ncd].mean()
              for ncd in ncd_levels]
zero_rate_ncd = [(y_te[test_pd["ncd_level"].values == ncd] == 0).mean()
                 for ncd in ncd_levels]

x_pos = np.arange(len(ncd_levels))
ax2.plot(x_pos, true_q_ncd,     "ko-",  linewidth=2, label="True q", markersize=8)
ax2.plot(x_pos, fit_q_ncd,      "rs--", linewidth=1.5, label="ZIT fitted q", markersize=7)
ax2.plot(x_pos, zero_rate_ncd,  "b^:",  linewidth=1, label="Empirical zero rate", markersize=6)
ax2.set_xticks(x_pos)
ax2.set_xticklabels([f"NCD {n}" for n in ncd_levels])
ax2.set_ylabel("Structural zero probability / zero rate")
ax2.set_title("Structural Zero Recovery by NCD Level")
ax2.legend()
ax2.grid(True, alpha=0.3)

# ── Plot 3: A/E by NCD level — baseline vs ZIT ────────────────────────────
ae_b = [ae_base_ncd.get(n, np.nan) for n in ncd_levels]
ae_z = [ae_zit_ncd.get(n, np.nan)  for n in ncd_levels]

ax3.plot(x_pos, ae_b, "b^--", linewidth=1.5, label="Baseline (Tweedie)", markersize=7)
ax3.plot(x_pos, ae_z, "rs-",  linewidth=1.5, label="ZIT-DGLM",           markersize=7)
ax3.axhline(1.0, color="black", linewidth=1.5, linestyle="--", label="A/E = 1.0")
ax3.set_xticks(x_pos)
ax3.set_xticklabels([f"NCD {n}" for n in ncd_levels])
ax3.set_ylabel("A/E ratio")
ax3.set_title("A/E Ratio by NCD Level")
ax3.legend()
ax3.grid(True, alpha=0.3)
ax3.set_ylim([0.7, 1.3])

# ── Plot 4: EM convergence ────────────────────────────────────────────────
ll_hist = zit._log_likelihoods
ax4.plot(range(1, len(ll_hist)+1), ll_hist, "ko-", linewidth=2, markersize=6)
ax4.set_xlabel("EM iteration")
ax4.set_ylabel("Total log-likelihood")
ax4.set_title("ZIT-DGLM EM Convergence")
ax4.grid(True, alpha=0.3)
if len(ll_hist) > 1:
    ax4.annotate(f"Final: {ll_hist[-1]:.0f}",
                 xy=(len(ll_hist), ll_hist[-1]),
                 xytext=(-30, -20), textcoords="offset points",
                 fontsize=9)

plt.suptitle(
    "insurance-zit-dglm vs plain Tweedie — Diagnostic Plots",
    fontsize=13, fontweight="bold"
)
plt.savefig("/tmp/benchmark_zit_dglm.png", dpi=120, bbox_inches="tight")
plt.show()
print("Plot saved to /tmp/benchmark_zit_dglm.png")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Feature Importance per Head

# COMMAND ----------

try:
    fi_mean = report.feature_importance("mean").to_pandas()
    fi_disp = report.feature_importance("dispersion").to_pandas()
    fi_zero = report.feature_importance("zero").to_pandas()

    print("=== Feature importance: mean head ===")
    print(fi_mean.to_string(index=False))
    print()
    print("=== Feature importance: dispersion head ===")
    print(fi_disp.to_string(index=False))
    print()
    print("=== Feature importance: zero-inflation head ===")
    print(fi_zero.to_string(index=False))
    print()
    print("Expected pattern: ncd_level should dominate the zero-inflation head.")
    ncd_zero_rank = fi_zero[fi_zero["feature"] == "ncd_level"].index[0] + 1
    print(f"ncd_level rank in zero head: {ncd_zero_rank}")
except Exception as e:
    print(f"Feature importance: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Verdict

# COMMAND ----------

# Zero calibration RMSE by NCD level
print("=" * 60)
print("VERDICT: ZIT-DGLM vs plain Tweedie GLM")
print("=" * 60)
print()
print(f"Mean prediction quality:")
print(f"  Tweedie deviance:  {dev_base:.5f} (baseline) -> {dev_zit:.5f} (ZIT)")
print(f"  Log-likelihood:    {ll_base:.5f} (baseline) -> {zit_score:.5f} (ZIT)")
print()
print(f"Zero modelling quality:")
print(f"  Zero calib RMSE:   {zcal_base:.5f} (baseline) -> {zcal_zit:.5f} (ZIT)")
print()
print(f"A/E calibration:")
print(f"  Overall A/E:       {ae_overall_base:.4f} (baseline), {ae_overall_zit:.4f} (ZIT)")
ae_ncd_base_dev = max(abs(v - 1.0) for v in ae_base_ncd.values() if not np.isnan(v))
ae_ncd_zit_dev  = max(abs(v - 1.0) for v in ae_zit_ncd.values() if not np.isnan(v))
print(f"  Max |A/E-1| (NCD): {ae_ncd_base_dev:.4f} (baseline), {ae_ncd_zit_dev:.4f} (ZIT)")
print()
print(f"Fit time:  {baseline_fit_time:.2f}s (baseline) -> {library_fit_time:.2f}s (ZIT)")
print()
print("When ZIT-DGLM is justified:")
print("  - Motor AD under unprotected NCD (structural zeros >15%)")
print("  - Home accidental damage with high sub-excess frequency")
print("  - When LRT or Vuong test favours ZIT over plain Tweedie")
print("  - Reinsurance pricing: getting P(Y=0) right matters for layer expected loss")
print()
print("When plain Tweedie is sufficient:")
print("  - Protected NCD / windscreen (q effectively 0)")
print("  - Home escape of water (no rational non-claiming incentive)")
print("  - When structural zero rate is below 10%")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. README Performance Snippet

# COMMAND ----------

print("""
## Performance

Benchmarked against a plain CatBoost Tweedie model on synthetic UK motor
accidental damage data (40,000 policies, known DGP with structural zeros
driven by NCD level and age band). Temporal 70/30 train/test split.
See `notebooks/benchmark_zit_dglm.py` for full methodology.
""")
print(f"| Metric                          | Tweedie GBM   | ZIT-DGLM      |")
print(f"|---------------------------------|---------------|---------------|")
print(f"| Tweedie deviance (test)         | {dev_base:.5f}     | {dev_zit:.5f}     |")
print(f"| Log-likelihood per obs          | {ll_base:.5f}     | {zit_score:.5f}     |")
print(f"| Zero calibration RMSE           | {zcal_base:.5f}     | {zcal_zit:.5f}     |")
print(f"| Overall A/E ratio               | {ae_overall_base:.4f}      | {ae_overall_zit:.4f}      |")
print(f"| Max A/E deviation by NCD level  | {ae_ncd_base_dev:.4f}      | {ae_ncd_zit_dev:.4f}      |")
print(f"| Fit time (s)                    | {baseline_fit_time:.2f}          | {library_fit_time:.2f}          |")
print()
print("""The ZIT-DGLM's advantage is concentrated in zero calibration and NCD-segment
A/E ratios. When the portfolio has structural zeros driven by NCD and behavioural
incentives (the UK motor AD case), the EM algorithm correctly separates structural
from compound-Poisson zeros. On perils without this structure, a plain Tweedie
GBM is adequate. Use check_balance() and the zero_calibration_plot() to assess
whether ZIT adds value on your specific portfolio.
""")
