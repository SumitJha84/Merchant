"""
health_score.py
Day 3 - Step 7
Reads:   data/processed/features_engineered.csv
         data/outputs/survival_predictions.csv
         data/processed/cluster_labels.csv
Produces: data/outputs/health_scores.csv
          data/outputs/shap_values.csv
          models/xgboost_health.pkl
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

import xgboost as xgb
import shap
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import joblib, os

os.makedirs("data/outputs", exist_ok=True)
os.makedirs("models", exist_ok=True)

SEED = 42

# ── load ──────────────────────────────────────────────────────────────────────
weekly  = pd.read_csv("data/processed/features_engineered.csv")
surv    = pd.read_csv("data/outputs/survival_predictions.csv")
clusters = pd.read_csv("data/processed/cluster_labels.csv")

# Get last-week features per merchant (static representation)
last_week = weekly.groupby("merchant_id").last().reset_index()

# Drop placeholder cols that will be filled by survival merge
for drop_col in ["survival_prob_4w", "revenue_at_risk", "health_score",
                 "cluster_label", "umap_x", "umap_y"]:
    if drop_col in last_week.columns:
        last_week = last_week.drop(columns=[drop_col])

# Merge survival and cluster
last_week = last_week.merge(surv[["merchant_id","survival_prob_4w","p_failure_4w","revenue_at_risk","net_revenue_at_risk"]], on="merchant_id", how="left")
cluster_last = clusters.groupby("merchant_id").last().reset_index()[["merchant_id","cluster_label","umap_x","umap_y"]]
last_week = last_week.merge(cluster_last, on="merchant_id", how="left")

print(f"Loaded {len(last_week)} merchants for health scoring")

# ── Build health score target (composite label) ────────────────────────────────
# No ground truth: construct a principled composite 0-100 target
# Higher = healthier

def build_health_target(df):
    from sklearn.preprocessing import MinMaxScaler
    mms = MinMaxScaler()

    s1 = df["survival_prob_4w"].fillna(0.5).values                      # 0-1, higher=healthy
    s2 = 1 - mms.fit_transform(df[["refund_rate"]].fillna(0))[:,0]      # 0-1
    s3 = 1 - mms.fit_transform(df[["chargeback_rate"]].fillna(0))[:,0]  # 0-1
    s4 = 1 - mms.fit_transform(df[["anomaly_score"]].fillna(0))[:,0]    # 0-1
    s5 = mms.fit_transform(df[["gmv_trend_slope"]].fillna(0))[:,0]      # 0-1

    raw = 0.35*s1 + 0.25*s2 + 0.20*s3 + 0.15*s4 + 0.05*s5
    score = (raw * 100).clip(0, 100)
    return score

last_week["health_score"] = build_health_target(last_week)

# ── XGBoost features ──────────────────────────────────────────────────────────
XGB_FEATURES = [
    "refund_rate", "chargeback_rate", "refund_acceleration",
    "chargeback_acceleration", "gmv_pct_change", "gmv_trend_slope",
    "gmv_volatility", "anomaly_score", "cb_to_refund_conversion",
    "refund_rate_zscore_peer", "chargeback_rate_zscore_peer",
    "refund_rate_zscore_personal", "gmv_consecutive_declines",
    "txn_count_volatility", "cumulative_refund_spikes",
    "refund_x_chargeback", "survival_prob_4w", "p_failure_4w",
    "merchant_vintage", "is_new_merchant", "is_top20_gmv_merchant",
    "gmv_mom_4w", "refund_jerk",
]

X = last_week[XGB_FEATURES].fillna(0).values
y = last_week["health_score"].values

# ── Train XGBoost ──────────────────────────────────────────────────────────────
model = xgb.XGBRegressor(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=SEED,
    n_jobs=-1,
    verbosity=0,
)
model.fit(X, y)

y_pred = model.predict(X)
rmse   = np.sqrt(mean_squared_error(y, y_pred))
print(f"XGBoost RMSE (train): {rmse:.4f}")

# ── SHAP Explainability ───────────────────────────────────────────────────────
explainer   = shap.TreeExplainer(model)
shap_values = explainer(pd.DataFrame(X, columns=XGB_FEATURES))

shap_df = pd.DataFrame(shap_values.values, columns=XGB_FEATURES)
shap_df.insert(0, "merchant_id", last_week["merchant_id"].values)

# Per-merchant top 3 drivers
def top_drivers(row, n=3):
    abs_vals = row.abs().sort_values(ascending=False)
    return [(feat, round(float(row[feat]), 4)) for feat in abs_vals.index[:n]]

shap_feature_cols = [c for c in shap_df.columns if c != "merchant_id"]
top3 = shap_df.apply(
    lambda r: top_drivers(r[shap_feature_cols]),
    axis=1
)
shap_df["top_driver_1"] = top3.apply(lambda x: x[0][0])
shap_df["top_driver_1_val"] = top3.apply(lambda x: x[0][1])
shap_df["top_driver_2"] = top3.apply(lambda x: x[1][0])
shap_df["top_driver_2_val"] = top3.apply(lambda x: x[1][1])
shap_df["top_driver_3"] = top3.apply(lambda x: x[2][0])
shap_df["top_driver_3_val"] = top3.apply(lambda x: x[2][1])

# Plain-English alert per merchant
def plain_english_alert(row):
    d1, v1 = row["top_driver_1"], row["top_driver_1_val"]
    d2, v2 = row["top_driver_2"], row["top_driver_2_val"]
    direction = lambda v: "increased risk" if v < 0 else "reduced risk"
    return (
        f"Score driven by {d1} ({direction(v1)}, {abs(v1):.2f}pts) "
        f"and {d2} ({direction(v2)}, {abs(v2):.2f}pts)."
    )

shap_df["alert_text"] = shap_df.apply(plain_english_alert, axis=1)

# Global feature importance (mean |SHAP|)
global_importance = pd.DataFrame({
    "feature": XGB_FEATURES,
    "mean_abs_shap": np.abs(shap_values.values).mean(axis=0)
}).sort_values("mean_abs_shap", ascending=False)

print("\nGlobal Feature Importance (mean |SHAP|):")
print(global_importance.head(10).to_string(index=False))

# ── Build final health scores output ──────────────────────────────────────────
health_out = last_week[[
    "merchant_id", "segment", "week",
    "health_score", "survival_prob_4w", "p_failure_4w",
    "revenue_at_risk", "net_revenue_at_risk",
    "refund_rate", "chargeback_rate", "anomaly_score",
    "cluster_label", "umap_x", "umap_y",
]].copy()

health_out["xgb_predicted_score"] = y_pred.round(2)

# Risk tier
def risk_tier(score):
    if score >= 75: return "Green"
    if score >= 50: return "Yellow"
    if score >= 25: return "Orange"
    return "Red"

health_out["risk_tier"] = health_out["health_score"].apply(risk_tier)
health_out = health_out.merge(
    shap_df[["merchant_id","top_driver_1","top_driver_1_val","top_driver_2","top_driver_2_val","alert_text"]],
    on="merchant_id", how="left"
)

# ── Weekly Risk Digest ────────────────────────────────────────────────────────
print("\n-- WEEKLY RISK DIGEST --------------------------------")
print(f"Total merchants scored : {len(health_out)}")
for tier in ["Red","Orange","Yellow","Green"]:
    sub = health_out[health_out["risk_tier"]==tier]
    rar = sub["revenue_at_risk"].sum()
    print(f"  {tier:8s}: {len(sub):3d} merchants | RaR: Rs {rar:>12,.0f}")

critical = health_out[health_out["risk_tier"]=="Red"].sort_values("revenue_at_risk", ascending=False)
print(f"\nTop 5 Critical Merchants:")
print(critical.head(5)[["merchant_id","segment","health_score","p_failure_4w","revenue_at_risk","alert_text"]].to_string(index=False))

# Early Warning Coverage Rate
# Of merchants that actually failed, what % are in Red/Orange?
from_survival = pd.read_csv("data/raw/merchant_survival_table.csv")
actual_failed = set(from_survival[from_survival["event"]==1]["merchant_id"])
flagged_risky = set(health_out[health_out["risk_tier"].isin(["Red","Orange"])]["merchant_id"])
coverage = len(actual_failed & flagged_risky) / len(actual_failed) if actual_failed else 0
print(f"\nEarly Warning Coverage Rate: {coverage:.1%}  (of {len(actual_failed)} failed merchants)")

# ── Save ──────────────────────────────────────────────────────────────────────
health_out.to_csv("data/outputs/health_scores.csv", index=False)
shap_df.to_csv("data/outputs/shap_values.csv", index=False)
global_importance.to_csv("data/outputs/shap_global_importance.csv", index=False)
joblib.dump(model, "models/xgboost_health.pkl")
joblib.dump(explainer, "models/shap_explainer.pkl")

print(f"\n[OK] health_scores.csv         -> {len(health_out)} merchants")
print(f"[OK] shap_values.csv           -> {len(shap_df)} rows, {len(shap_feature_cols)} features")
print(f"[OK] shap_global_importance.csv saved")
print(f"[OK] xgboost_health.pkl + shap_explainer.pkl saved")
