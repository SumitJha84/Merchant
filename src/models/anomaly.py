"""
anomaly.py
Day 2 - Step 4
Reads:   data/processed/features_engineered.csv
Produces: data/processed/anomaly_scores.csv
          Updates anomaly_score col in features_engineered.csv

Two-layer anomaly detection:
  Layer 1 - IsolationForest (global population anomaly)
  Layer 2 - Per-merchant rolling z-score (already in features, aggregated here)
  Final    - Ensemble score (weighted average)
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib
import os

os.makedirs("data/processed", exist_ok=True)
os.makedirs("models", exist_ok=True)

SEED = 42

# ── load ──────────────────────────────────────────────────────────────────────
df = pd.read_csv("data/processed/features_engineered.csv")
df = df.sort_values(["merchant_id", "week"]).reset_index(drop=True)
print(f"Loaded {len(df):,} rows x {len(df.columns)} cols")

# ── feature set for IsolationForest ──────────────────────────────────────────
IF_FEATURES = [
    "refund_rate",
    "chargeback_rate",
    "gmv_pct_change",
    "txn_count_volatility",
    "refund_acceleration",
    "chargeback_acceleration",
    "cb_to_refund_conversion",
    "refund_x_chargeback",
    "gmv_consecutive_declines",
    "refund_rate_zscore_peer",
    "chargeback_rate_zscore_peer",
]

X = df[IF_FEATURES].values

# ── scale ─────────────────────────────────────────────────────────────────────
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ── Layer 1: IsolationForest ──────────────────────────────────────────────────
iso = IsolationForest(
    n_estimators=200,
    contamination=0.15,
    max_samples="auto",
    random_state=SEED,
    n_jobs=-1,
)
iso.fit(X_scaled)

# Raw scores: more negative = more anomalous
raw_scores = iso.score_samples(X_scaled)          # range ~ [-0.7, 0.1]
# Normalize to [0, 100]: higher = more anomalous
iso_score_norm = (raw_scores - raw_scores.max()) / (raw_scores.min() - raw_scores.max() + 1e-9)
df["if_anomaly_score"] = (iso_score_norm * 100).clip(0, 100)

# ── Layer 2: Personal z-score composite ──────────────────────────────────────
# Combine personal z-scores into a single "personal anomaly" score
personal_z = (
    df["refund_rate_zscore_personal"].clip(0, 5) / 5 * 0.5 +
    df["chargeback_rate_zscore_personal"].clip(0, 5) / 5 * 0.3 +
    df["gmv_zscore_personal"].clip(-5, 0).abs() / 5 * 0.2
)
df["personal_anomaly_score"] = (personal_z * 100).clip(0, 100)

# ── Ensemble: weighted combination ───────────────────────────────────────────
# Global IF catches population-level outliers
# Personal score catches merchant-specific deterioration
df["anomaly_score"] = (
    0.55 * df["if_anomaly_score"] +
    0.45 * df["personal_anomaly_score"]
).clip(0, 100).round(2)

# ── Anomaly flag ──────────────────────────────────────────────────────────────
df["is_anomalous"] = (df["anomaly_score"] > 70).astype(int)

# ── Precision@K validation (using segment as proxy ground truth) ───────────────
# "Fraudulent" and "risky" = positive class
df["true_risky"] = df["segment"].isin(["fraudulent", "risky"]).astype(int)

# Sort by anomaly score, check top-150 (30% of 500 merchants × any week sample)
week12 = df[df["week"] == 12].copy()
week12_sorted = week12.sort_values("anomaly_score", ascending=False)
k = 50
top_k = week12_sorted.head(k)
precision_k = top_k["true_risky"].mean()
print(f"\n[Validation] Precision@{k} at week 12: {precision_k:.3f}")

# By-segment mean anomaly score
print("\nMean anomaly_score by segment (week 12):")
print(week12.groupby("segment")["anomaly_score"].mean().round(2).sort_values(ascending=False))

# ── Save models ───────────────────────────────────────────────────────────────
joblib.dump(iso, "models/isolation_forest.pkl")
joblib.dump(scaler, "models/if_scaler.pkl")

# ── Output: anomaly_scores.csv (lean — key columns only) ──────────────────────
anomaly_out = df[[
    "merchant_id", "week", "segment",
    "if_anomaly_score", "personal_anomaly_score",
    "anomaly_score", "is_anomalous",
    "refund_spike_flag", "cumulative_refund_spikes",
]].copy()
anomaly_out.to_csv("data/processed/anomaly_scores.csv", index=False)

# ── Update features_engineered.csv with final anomaly_score ───────────────────
df.to_csv("data/processed/features_engineered.csv", index=False)

print(f"\n[OK] anomaly_scores.csv    -> {len(anomaly_out):,} rows")
print(f"[OK] Anomalous weeks flagged: {df['is_anomalous'].sum()} / {len(df)}")
print(f"[OK] Models saved: isolation_forest.pkl, if_scaler.pkl")
print(f"\nAnomaly score distribution:")
print(df["anomaly_score"].describe().round(2))
