"""
engineer_features.py
Day 2 - Step 3
Reads:   data/raw/merchant_weekly_features.csv
Produces: data/processed/features_engineered.csv

Differentiators added:
- Per-merchant rolling z-score anomaly (personalized, not global)
- Refund velocity spike detector (3-sigma above own baseline)
- Chargeback-to-refund conversion rate
- Merchant vintage effect
- Peer-segment z-score for refund_rate
"""

import numpy as np
import pandas as pd
from scipy import stats
import os

os.makedirs("data/processed", exist_ok=True)

# ── load ──────────────────────────────────────────────────────────────────────
df = pd.read_csv("data/raw/merchant_weekly_features.csv")
df = df.sort_values(["merchant_id", "week"]).reset_index(drop=True)
print(f"Loaded {len(df):,} rows")

grp = df.groupby("merchant_id")

# ── 1. Merchant vintage (weeks since onboarding = week number itself) ─────────
# In simulation all merchants start week 1, so vintage = week
df["merchant_vintage"] = df["week"]
df["is_new_merchant"]  = (df["week"] <= 4).astype(int)

# ── 2. Per-merchant rolling z-score (personalized anomaly) ────────────────────
# z = (value - rolling_mean) / (rolling_std + eps)  over 8-week window
for col in ["refund_rate", "chargeback_rate", "gmv"]:
    roll_mean = grp[col].transform(lambda x: x.rolling(8, min_periods=3).mean())
    roll_std  = grp[col].transform(lambda x: x.rolling(8, min_periods=3).std().fillna(0))
    df[f"{col}_zscore_personal"] = (df[col] - roll_mean) / (roll_std + 1e-6)

# ── 3. Refund velocity spike detector ─────────────────────────────────────────
# Flag when refund_rate crosses 3σ above own 8-week rolling baseline
df["refund_spike_flag"] = (df["refund_rate_zscore_personal"] > 3.0).astype(int)

# Cumulative spikes per merchant up to current week
df["cumulative_refund_spikes"] = grp["refund_spike_flag"].transform("cumsum")

# ── 4. Chargeback-to-refund conversion rate ───────────────────────────────────
# High = systematic fraud / catastrophic ops failure
df["cb_to_refund_conversion"] = df["chargeback_count"] / (df["refund_count"] + 1e-6)
df["cb_to_refund_conversion"]  = df["cb_to_refund_conversion"].clip(0, 1)

# ── 5. Peer-segment z-score (cross-merchant, within segment per week) ──────────
# "Is merchant X's refund rate abnormal for its segment this week?"
for col in ["refund_rate", "chargeback_rate", "gmv_pct_change"]:
    seg_week_mean = df.groupby(["segment", "week"])[col].transform("mean")
    seg_week_std  = df.groupby(["segment", "week"])[col].transform("std").fillna(1e-6)
    df[f"{col}_zscore_peer"] = (df[col] - seg_week_mean) / (seg_week_std + 1e-6)

# ── 6. GMV concentration risk flag ───────────────────────────────────────────
# Flag merchants in top 20% of GMV (Pareto — they carry disproportionate risk)
gmv_per_merchant = df.groupby("merchant_id")["gmv"].mean()
p80 = gmv_per_merchant.quantile(0.80)
top_gmv_merchants = set(gmv_per_merchant[gmv_per_merchant >= p80].index)
df["is_top20_gmv_merchant"] = df["merchant_id"].isin(top_gmv_merchants).astype(int)

# ── 7. GMV momentum features ──────────────────────────────────────────────────
df["gmv_mom_2w"] = grp["gmv"].transform(lambda x: x.pct_change(2).fillna(0))
df["gmv_mom_4w"] = grp["gmv"].transform(lambda x: x.pct_change(4).fillna(0))

# Consecutive declining weeks
def consecutive_declines(series):
    result = []
    count = 0
    prev = None
    for val in series:
        if prev is not None and val < prev:
            count += 1
        else:
            count = 0
        result.append(count)
        prev = val
    return result

df["gmv_consecutive_declines"] = grp["gmv"].transform(
    lambda x: pd.Series(consecutive_declines(x.values), index=x.index)
)

# ── 8. Refund / chargeback trend features ────────────────────────────────────
df["refund_rate_8w_avg"]  = grp["refund_rate"].transform(lambda x: x.rolling(8, min_periods=2).mean())
df["refund_rate_8w_std"]  = grp["refund_rate"].transform(lambda x: x.rolling(8, min_periods=2).std().fillna(0))

# Rate of change of acceleration (jerk) — third derivative proxy
df["refund_jerk"] = grp["refund_acceleration"].transform(lambda x: x.diff().fillna(0))

# ── 9. Net revenue proxy ──────────────────────────────────────────────────────
# PayU earns ~1.8-2.2% MDR on transactions
MDR_RATE = 0.020
df["estimated_fee_revenue"]  = df["gmv"] * MDR_RATE
df["net_gmv_after_refunds"]  = df["gmv"] - df["refund_amount"]
df["net_revenue_at_risk_est"] = df["net_gmv_after_refunds"] * (df["chargeback_rate"] + df["refund_rate"])

# ── 10. Settlement window exposure ────────────────────────────────────────────
# PayU holds funds for ~7 days; approximate as single-week GMV
df["settlement_exposure"] = df["gmv"]  # 1 week of GMV = unsettled exposure

# ── 11. Interaction features (for XGBoost) ───────────────────────────────────
df["refund_x_chargeback"]    = df["refund_rate"] * df["chargeback_rate"]
df["anomaly_x_gmv_decline"]  = df["gmv_pct_change"].clip(-1, 0).abs() * df["refund_rate_zscore_personal"].clip(0)

# ── 12. Lagged features (week t-1, t-2) ──────────────────────────────────────
for col in ["refund_rate", "chargeback_rate", "gmv_pct_change"]:
    df[f"{col}_lag1"] = grp[col].transform(lambda x: x.shift(1).fillna(0))
    df[f"{col}_lag2"] = grp[col].transform(lambda x: x.shift(2).fillna(0))

# ── 13. Time-to-end features ─────────────────────────────────────────────────
df["weeks_remaining"] = 24 - df["week"]

# ── cleanup: fill residual NaNs ──────────────────────────────────────────────
num_cols = df.select_dtypes(include=[np.number]).columns
df[num_cols] = df[num_cols].fillna(0)

# ── save ──────────────────────────────────────────────────────────────────────
out_path = "data/processed/features_engineered.csv"
df.to_csv(out_path, index=False)

print(f"[OK] features_engineered.csv -> {len(df):,} rows x {len(df.columns)} columns")
print(f"\nNew features added: {len(df.columns)} total columns")
print(f"\nRefund spike events: {df['refund_spike_flag'].sum()} ({df['refund_spike_flag'].mean()*100:.1f}% of weeks)")
print(f"Top-20% GMV merchants: {df['is_top20_gmv_merchant'].sum() // 24}")
print(f"\nSample engineered features:")
print(df[["merchant_id","week","refund_rate_zscore_personal","refund_spike_flag",
          "cb_to_refund_conversion","refund_rate_zscore_peer"]].head(10).to_string(index=False))
