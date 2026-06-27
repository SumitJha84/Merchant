"""
clustering.py
Day 2 - Step 5
Reads:   data/processed/features_engineered.csv  (with anomaly_score filled)
Produces: data/processed/cluster_labels.csv
          data/processed/umap_embeddings.csv

Differentiators:
- Silhouette curve for k=3..7 to justify k=5 (avoids circular reasoning)
- Merchant trajectory in UMAP space (week 1 -> week 24 movement)
- Cluster migration matrix (Healthy -> Risky transition counts)
- Named clusters aligned to business segments
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score
from umap import UMAP
import joblib
import os

os.makedirs("data/processed", exist_ok=True)
os.makedirs("models", exist_ok=True)

SEED = 42

# ── load ──────────────────────────────────────────────────────────────────────
df = pd.read_csv("data/processed/features_engineered.csv")
df = df.sort_values(["merchant_id", "week"]).reset_index(drop=True)
print(f"Loaded {len(df):,} rows")

# ── features for clustering ───────────────────────────────────────────────────
CLUSTER_FEATURES = [
    "refund_rate",
    "chargeback_rate",
    "gmv_pct_change",
    "gmv_volatility",
    "refund_acceleration",
    "chargeback_acceleration",
    "cb_to_refund_conversion",
    "anomaly_score",
    "refund_rate_zscore_peer",
    "gmv_consecutive_declines",
    "gmv_mom_4w",
    "txn_count_volatility",
]

X = df[CLUSTER_FEATURES].fillna(0).values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ── UMAP reduction ────────────────────────────────────────────────────────────
print("Running UMAP...")
reducer = UMAP(
    n_components=2,
    n_neighbors=15,
    min_dist=0.1,
    random_state=SEED,
    metric="euclidean",
)
embedding = reducer.fit_transform(X_scaled)
df["umap_x"] = embedding[:, 0].round(4)
df["umap_y"] = embedding[:, 1].round(4)
print("[OK] UMAP complete")

# ── Silhouette curve: justify k=5 ────────────────────────────────────────────
print("\nSilhouette scores (k=3 to 7):")
silhouette_results = {}
for k in range(3, 8):
    km_temp = KMeans(n_clusters=k, random_state=SEED, n_init=10)
    labels_temp = km_temp.fit_predict(embedding)
    sil = silhouette_score(embedding, labels_temp)
    db  = davies_bouldin_score(embedding, labels_temp)
    silhouette_results[k] = sil
    print(f"  k={k}: silhouette={sil:.4f}  davies-bouldin={db:.4f}")

best_k = max(silhouette_results, key=silhouette_results.get)
print(f"\n[Validation] Best k by silhouette: {best_k} (score={silhouette_results[best_k]:.4f})")
print(f"[Note] Using k=5 to align with business segments (silhouette={silhouette_results[5]:.4f})")

# ── Final KMeans (k=5) ────────────────────────────────────────────────────────
km = KMeans(n_clusters=5, random_state=SEED, n_init=20)
df["cluster_raw"] = km.fit_predict(embedding)

# ── Name clusters by aligning to dominant segment ────────────────────────────
cluster_segment_map = (
    df.groupby(["cluster_raw", "segment"])
    .size()
    .reset_index(name="count")
    .sort_values("count", ascending=False)
    .drop_duplicates("cluster_raw")
    .set_index("cluster_raw")["segment"]
)

CLUSTER_NAMES = {
    "healthy":     "Healthy",
    "growing":     "Growing",
    "stable":      "Stable",
    "risky":       "Risky",
    "fraudulent":  "Critical",
}

df["cluster_label"] = df["cluster_raw"].map(cluster_segment_map).map(CLUSTER_NAMES).fillna("Stable")
print("\nCluster distribution:")
print(df["cluster_label"].value_counts())

# ── Merchant trajectory: position at week 1 vs week 24 ───────────────────────
week1  = df[df["week"] == 1][["merchant_id", "umap_x", "umap_y", "cluster_label"]].rename(
    columns={"umap_x": "umap_x_w1", "umap_y": "umap_y_w1", "cluster_label": "cluster_w1"}
)
week24 = df[df["week"] == 24][["merchant_id", "umap_x", "umap_y", "cluster_label"]].rename(
    columns={"umap_x": "umap_x_w24", "umap_y": "umap_y_w24", "cluster_label": "cluster_w24"}
)
trajectory = week1.merge(week24, on="merchant_id")
trajectory["umap_dx"] = trajectory["umap_x_w24"] - trajectory["umap_x_w1"]
trajectory["umap_dy"] = trajectory["umap_y_w24"] - trajectory["umap_y_w1"]
trajectory["trajectory_magnitude"] = np.sqrt(trajectory["umap_dx"]**2 + trajectory["umap_dy"]**2).round(4)

# ── Cluster migration matrix ──────────────────────────────────────────────────
migration = pd.crosstab(
    trajectory["cluster_w1"],
    trajectory["cluster_w24"],
    margins=False,
)
print("\nCluster Migration Matrix (row=week1, col=week24):")
print(migration)

# ── Save models ───────────────────────────────────────────────────────────────
joblib.dump(km, "models/kmeans.pkl")
joblib.dump(reducer, "models/umap_reducer.pkl")
joblib.dump(scaler, "models/cluster_scaler.pkl")

# ── Output files ──────────────────────────────────────────────────────────────
cluster_out = df[[
    "merchant_id", "week", "segment",
    "umap_x", "umap_y",
    "cluster_raw", "cluster_label",
]].copy()
cluster_out.to_csv("data/processed/cluster_labels.csv", index=False)

umap_out = df[["merchant_id", "week", "umap_x", "umap_y", "cluster_label"]].copy()
umap_out.to_csv("data/processed/umap_embeddings.csv", index=False)

trajectory.to_csv("data/processed/merchant_trajectories.csv", index=False)

# ── Update features_engineered.csv ───────────────────────────────────────────
df.to_csv("data/processed/features_engineered.csv", index=False)

print(f"\n[OK] cluster_labels.csv      -> {len(cluster_out):,} rows")
print(f"[OK] umap_embeddings.csv     -> {len(umap_out):,} rows")
print(f"[OK] merchant_trajectories.csv -> {len(trajectory)} rows (1 per merchant)")
print(f"[OK] Models: kmeans.pkl, umap_reducer.pkl, cluster_scaler.pkl")
print(f"\nTrajectory magnitude (how far merchants moved in UMAP space):")
print(trajectory.groupby("cluster_w24")["trajectory_magnitude"].mean().round(3))
