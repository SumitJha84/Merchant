"""
master_builder.py — Day 3 Step 8
Reads all model outputs, adds drift scores + watchlist, writes master_dataframe.csv
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import os

os.makedirs("data/outputs", exist_ok=True)

weekly   = pd.read_csv("data/processed/features_engineered.csv")
health   = pd.read_csv("data/outputs/health_scores.csv")
survival = pd.read_csv("data/outputs/survival_predictions.csv")
shap_df  = pd.read_csv("data/outputs/shap_values.csv")
weekly   = weekly.sort_values(["merchant_id","week"]).reset_index(drop=True)

DRIFT_FEATURES = [
    "refund_rate","chargeback_rate","gmv_pct_change","gmv_volatility",
    "refund_acceleration","chargeback_acceleration","cb_to_refund_conversion",
    "anomaly_score","txn_count_volatility",
]

sc = StandardScaler()
X  = sc.fit_transform(weekly[DRIFT_FEATURES].replace([np.inf,-np.inf],np.nan).fillna(0))
weekly[DRIFT_FEATURES] = X

w1 = weekly[weekly["week"]==1].set_index("merchant_id")[DRIFT_FEATURES]
wN = weekly.groupby("merchant_id")[DRIFT_FEATURES].last()
idx = w1.index.intersection(wN.index)
v1, vN = w1.loc[idx].values, wN.loc[idx].values

cos_sim   = np.array([cosine_similarity(v1[i:i+1], vN[i:i+1])[0,0] for i in range(len(idx))])
euc_dist  = np.linalg.norm(v1 - vN, axis=1)
drift = pd.DataFrame({
    "merchant_id":     idx,
    "drift_score":     np.clip(1-cos_sim, 0, 1).round(4),
    "drift_euclidean": (euc_dist / (euc_dist.max()+1e-9)).round(4),
})
drift["drift_composite"] = (0.6*drift["drift_score"] + 0.4*drift["drift_euclidean"]).round(4)

master = health.copy()
master = master.merge(survival[["merchant_id","survival_prob_now","gmv_4w_avg"]], on="merchant_id", how="left")
master = master.merge(drift, on="merchant_id", how="left")
master = master.merge(
    shap_df[["merchant_id","top_driver_1","top_driver_1_val","top_driver_2","top_driver_2_val","alert_text"]],
    on="merchant_id", how="left"
)

master["watchlist_flag"] = (
    (master["health_score"] < 50) |
    (master["drift_composite"] > 0.4) |
    (master["p_failure_4w"] > 0.6)
).astype(int)

def severity(r):
    if r["health_score"] < 25 or r["p_failure_4w"] > 0.75: return "CRITICAL"
    if r["health_score"] < 40 or r["drift_composite"] > 0.5: return "HIGH"
    if r["health_score"] < 55 or r["drift_composite"] > 0.35: return "MEDIUM"
    return "LOW"
master["alert_severity"] = master.apply(severity, axis=1)

cluster_rar = master.groupby("cluster_label").agg(
    merchant_count=("merchant_id","count"),
    total_rar=("revenue_at_risk","sum"),
    avg_health=("health_score","mean"),
).reset_index()
cluster_rar.to_csv("data/outputs/cluster_rar.csv", index=False)

print(f"Portfolio RaR : Rs {master['revenue_at_risk'].sum():,.0f}")
print(f"Avg Health    : {master['health_score'].mean():.1f}")
print(f"Critical      : {(master['alert_severity']=='CRITICAL').sum()}")
print(f"Watchlist     : {master['watchlist_flag'].sum()}")
print(f"High Drift    : {(master['drift_composite']>0.4).sum()}")

master.to_csv("data/outputs/master_dataframe.csv", index=False)
print(f"[OK] master_dataframe.csv -> {len(master)} rows x {len(master.columns)} cols")
