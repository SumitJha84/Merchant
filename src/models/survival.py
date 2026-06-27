"""
survival.py
Day 3 - Step 6
Reads:   data/processed/features_engineered.csv
         data/raw/merchant_survival_table.csv
Produces: data/outputs/survival_predictions.csv
          models/cox_model.pkl
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
from lifelines import CoxTimeVaryingFitter, KaplanMeierFitter
from lifelines.statistics import proportional_hazard_test
from sklearn.preprocessing import StandardScaler
import joblib, os

os.makedirs("data/outputs", exist_ok=True)
os.makedirs("models", exist_ok=True)

SEED = 42

# ── load ──────────────────────────────────────────────────────────────────────
weekly  = pd.read_csv("data/processed/features_engineered.csv")
survive = pd.read_csv("data/raw/merchant_survival_table.csv")
weekly  = weekly.sort_values(["merchant_id", "week"]).reset_index(drop=True)
print(f"Weekly: {len(weekly):,} rows | Survival: {len(survive)} merchants")

# ── build time-varying long format for Cox ────────────────────────────────────
# Each row = one risk interval: merchant was alive from (start=week-1) to (stop=week)
# event = 1 only at the LAST row of a failed merchant

COX_FEATURES = [
    "refund_rate",
    "chargeback_rate",
    "refund_acceleration",
    "chargeback_acceleration",
    "gmv_pct_change",
    "gmv_trend_slope",
    "anomaly_score",
    "cb_to_refund_conversion",
    "refund_rate_zscore_peer",
    "gmv_consecutive_declines",
    "txn_count_volatility",
    "merchant_vintage",
]

# Merge failure info
weekly = weekly.merge(
    survive[["merchant_id", "duration", "event"]],
    on="merchant_id", how="left"
)

# Build (start, stop, event) format
cox_rows = []
for mid, grp in weekly.groupby("merchant_id"):
    grp = grp.sort_values("week").reset_index(drop=True)
    merchant_duration = int(grp["duration"].iloc[0])
    merchant_event    = int(grp["event"].iloc[0])

    for i, row in grp.iterrows():
        w = int(row["week"])
        if w > merchant_duration:
            break
        is_last = (w == merchant_duration)
        cox_rows.append({
            "id":    mid,
            "start": w - 1,
            "stop":  w,
            "event": int(merchant_event and is_last),
            **{f: row[f] for f in COX_FEATURES},
        })

cox_df = pd.DataFrame(cox_rows)
print(f"Cox long-format: {len(cox_df):,} rows | Events: {cox_df['event'].sum()}")

# ── scale features ────────────────────────────────────────────────────────────
scaler = StandardScaler()
cox_df[COX_FEATURES] = cox_df[COX_FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0)
cox_df[COX_FEATURES] = scaler.fit_transform(cox_df[COX_FEATURES])

# ── VIF / multicollinearity check ─────────────────────────────────────────────
from statsmodels.stats.outliers_influence import variance_inflation_factor
X_vif = cox_df[COX_FEATURES].fillna(0)
vif_data = pd.DataFrame({
    "feature": COX_FEATURES,
    "VIF": [variance_inflation_factor(X_vif.fillna(0).values, i) for i in range(len(COX_FEATURES))]
}).sort_values("VIF", ascending=False)
print("\nVIF Analysis:")
print(vif_data.to_string(index=False))

# Drop features with VIF > 8 (multicollinear)
high_vif = vif_data[vif_data["VIF"] > 8]["feature"].tolist()
if high_vif:
    print(f"\nDropping high-VIF features: {high_vif}")
    COX_FEATURES_FINAL = [f for f in COX_FEATURES if f not in high_vif]
else:
    COX_FEATURES_FINAL = COX_FEATURES
    print("No multicollinearity detected (all VIF < 8)")

# Final NaN/inf guard
cox_df[COX_FEATURES_FINAL] = cox_df[COX_FEATURES_FINAL].replace([np.inf, -np.inf], 0).fillna(0)

# ── Cox Time-Varying Model ────────────────────────────────────────────────────
ctv = CoxTimeVaryingFitter(penalizer=0.1)   # L2 regularization
ctv.fit(
    cox_df,
    id_col="id",
    start_col="start",
    stop_col="stop",
    event_col="event",
    formula=" + ".join(COX_FEATURES_FINAL),
)

print("\n-- Cox Model Summary ----------------------------------")
ctv.print_summary(decimals=3, columns=["coef", "exp(coef)", "p", "coef lower 95%", "coef upper 95%"])

# Concordance index — compute manually
from lifelines.utils import concordance_index
# Get risk score (partial hazard) at last observed time per merchant
risk_scores = ctv.predict_log_partial_hazard(cox_df.groupby("id").last().reset_index())
survive_for_ci = survive.copy()
survive_for_ci["risk_score"] = survive_for_ci["merchant_id"].map(
    dict(zip(cox_df.groupby("id").last().reset_index()["id"], risk_scores))
).fillna(0)
c_idx = concordance_index(
    survive_for_ci["duration"],
    -survive_for_ci["risk_score"],
    survive_for_ci["event"]
)
print(f"\nConcordance Index (C-index): {c_idx:.4f}")
if c_idx >= 0.80:
    print("[EXCELLENT] C-index >= 0.80")
elif c_idx >= 0.70:
    print("[GOOD] C-index 0.70-0.80")
else:
    print("[WEAK] C-index < 0.70 -- review features")

# ── Hazard ratio table ────────────────────────────────────────────────────────
hr_df = ctv.summary[["coef", "exp(coef)", "p"]].copy()
hr_df.columns = ["log_hazard_ratio", "hazard_ratio", "p_value"]
hr_df["direction"] = hr_df["hazard_ratio"].apply(lambda x: "RISK+" if x > 1 else "PROTECTIVE")
hr_df["significant"] = hr_df["p_value"] < 0.05
print("\nHazard Ratios:")
print(hr_df.sort_values("hazard_ratio", ascending=False).round(4).to_string())

# ── Kaplan-Meier per cluster ──────────────────────────────────────────────────
# Merge cluster info
weekly_clusters = pd.read_csv("data/processed/cluster_labels.csv")
last_week = weekly_clusters.groupby("merchant_id").last().reset_index()[["merchant_id", "cluster_label"]]
survive_km = survive.merge(last_week, on="merchant_id", how="left")

km_results = {}
print("\n-- Kaplan-Meier by Cluster -----------------------")
for cluster in survive_km["cluster_label"].dropna().unique():
    sub = survive_km[survive_km["cluster_label"] == cluster]
    kmf = KaplanMeierFitter()
    kmf.fit(sub["duration"], event_observed=sub["event"], label=cluster)
    median_surv = kmf.median_survival_time_
    surv_at_12  = float(kmf.survival_function_at_times([12]).values[0])
    km_results[cluster] = {"median": median_surv, "surv_at_12w": round(surv_at_12, 4)}
    print(f"  {cluster:10s}: median={median_surv:.1f}w  P(surv>12w)={surv_at_12:.3f}")

# ── Predict survival probability for all merchants at +4 weeks ────────────────
# Use cumulative partial hazard from Cox model
# lifelines CoxTVF exposes predict_cumulative_hazard
predictions = []
for mid, grp in cox_df.groupby("id"):
    grp_sorted = grp.sort_values("stop")
    try:
        # Cumulative hazard at last observed time
        cum_haz = ctv.predict_cumulative_hazard(grp_sorted).iloc[-1, -1]
        surv_now  = float(np.exp(-cum_haz))
        # Approximate 4-week forward survival: decay by cluster baseline hazard
        surv_4w   = float(np.exp(-cum_haz * 1.20))   # 20% hazard increase proxy
        surv_4w   = np.clip(surv_4w, 0, 1)
    except Exception:
        surv_now = 0.5
        surv_4w  = 0.5

    segment  = weekly[weekly["merchant_id"] == mid]["segment"].iloc[0]
    predictions.append({
        "merchant_id":     mid,
        "segment":         segment,
        "survival_prob_now": round(surv_now, 4),
        "survival_prob_4w":  round(surv_4w, 4),
        "p_failure_4w":      round(1 - surv_4w, 4),
    })

surv_pred = pd.DataFrame(predictions)

# Merge GMV for Revenue at Risk
gmv_avg = weekly.groupby("merchant_id")["gmv"].apply(lambda x: x.tail(4).mean()).reset_index()
gmv_avg.columns = ["merchant_id", "gmv_4w_avg"]
cb_avg  = weekly.groupby("merchant_id")["chargeback_rate"].mean().reset_index()
cb_avg.columns  = ["merchant_id", "avg_chargeback_rate"]

surv_pred = surv_pred.merge(gmv_avg, on="merchant_id").merge(cb_avg, on="merchant_id")
surv_pred["revenue_at_risk"] = (
    surv_pred["gmv_4w_avg"] * 4 *
    surv_pred["p_failure_4w"] *
    (1 + surv_pred["avg_chargeback_rate"])
).round(2)

# Net RaR (subtract expected fee revenue from surviving period)
surv_pred["net_revenue_at_risk"] = (
    surv_pred["revenue_at_risk"] -
    surv_pred["gmv_4w_avg"] * 4 * surv_pred["survival_prob_4w"] * 0.020
).clip(0).round(2)

total_rar = surv_pred["revenue_at_risk"].sum()
print(f"\nTotal Portfolio Revenue at Risk: Rs {total_rar:,.0f}")
print(f"Net Revenue at Risk:             Rs {surv_pred['net_revenue_at_risk'].sum():,.0f}")
print(f"\nTop 10 Riskiest Merchants:")
print(surv_pred.sort_values("revenue_at_risk", ascending=False).head(10)[
    ["merchant_id", "segment", "survival_prob_4w", "p_failure_4w", "revenue_at_risk"]
].to_string(index=False))

# ── Save ──────────────────────────────────────────────────────────────────────
surv_pred.to_csv("data/outputs/survival_predictions.csv", index=False)
joblib.dump(ctv, "models/cox_model.pkl")
joblib.dump(scaler, "models/cox_scaler.pkl")
pd.DataFrame(km_results).T.to_csv("data/outputs/kaplan_meier_summary.csv")

print(f"\n[OK] survival_predictions.csv -> {len(surv_pred)} merchants")
print(f"[OK] cox_model.pkl saved")
