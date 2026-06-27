"""
simulate_merchants.py
Day 1 — Step 1
Produces: data/raw/merchant_weekly_features.csv (12,000 rows)
"""

import numpy as np
import pandas as pd
from scipy import stats
import os

SEED = 42
rng = np.random.default_rng(SEED)

N_MERCHANTS = 500
N_WEEKS = 24
FESTIVAL_WEEKS = {10: 0.25, 18: 0.20, 22: 0.40}  # week -> GMV uplift

SEGMENTS = {
    "healthy":     150,
    "growing":     100,
    "stable":      100,
    "risky":       100,
    "fraudulent":   50,
}

# ── helpers ──────────────────────────────────────────────────────────────────

def assign_segments():
    ids, segs = [], []
    i = 1
    for seg, count in SEGMENTS.items():
        for _ in range(count):
            ids.append(f"MID_{i:04d}")
            segs.append(seg)
            i += 1
    df = pd.DataFrame({"merchant_id": ids, "segment": segs})
    return df.sample(frac=1, random_state=SEED).reset_index(drop=True)


def base_gmv(segment):
    """Log-normal base weekly GMV per segment (₹)."""
    params = {
        "healthy":    (12.0, 0.50),
        "growing":    (11.5, 0.60),
        "stable":     (11.8, 0.45),
        "risky":      (11.2, 0.70),
        "fraudulent": (12.5, 0.80),
    }
    mu, sigma = params[segment]
    return float(rng.lognormal(mu, sigma))


def seasonal_params():
    A = float(rng.uniform(0.05, 0.15))
    B = float(rng.uniform(0.02, 0.08))
    return A, B


def seasonal_multiplier(week, A, B):
    return 1 + A * np.sin(2 * np.pi * week / 52) + B * np.cos(2 * np.pi * week / 52)


def festival_multiplier(week):
    return 1 + FESTIVAL_WEEKS.get(week, 0.0)


def base_refund_rate(segment):
    """Beta(2,20) mean ~0.09; segment shifts applied."""
    base = float(rng.beta(2, 20))
    offsets = {"healthy": -0.02, "growing": -0.01, "stable": 0.0, "risky": 0.02, "fraudulent": 0.05}
    return np.clip(base + offsets[segment], 0.01, 0.70)


def base_chargeback_rate(segment):
    """Beta(1,50) mean ~0.02; segment shifts applied."""
    base = float(rng.beta(1, 50))
    offsets = {"healthy": 0.0, "growing": -0.002, "stable": 0.0, "risky": 0.01, "fraudulent": 0.03}
    return np.clip(base + offsets[segment], 0.001, 0.30)


def txn_count_from_gmv(gmv, segment):
    """Derive transaction count from GMV using segment avg ticket."""
    avg_tickets = {"healthy": 1200, "growing": 950, "stable": 1100, "risky": 800, "fraudulent": 600}
    ticket = avg_tickets[segment] * rng.uniform(0.85, 1.15)
    return max(1, int(gmv / ticket))


# ── per-merchant weekly simulation ───────────────────────────────────────────

def simulate_merchant(merchant_id, segment, failure_week):
    A, B = seasonal_params()
    gmv_base = base_gmv(segment)
    refund_r = base_refund_rate(segment)
    charge_r = base_chargeback_rate(segment)

    # Growth / decay rates
    growth_rate = float(rng.uniform(0.01, 0.04)) if segment == "growing" else 0.0
    decay_rate  = float(rng.uniform(0.02, 0.06)) if segment == "risky" else 0.0

    rows = []
    prev_gmv = None
    prev_refund_r = refund_r
    refund_history = []  # for chargeback lag

    for week in range(1, N_WEEKS + 1):

        # ── post-failure: zero out ──
        if failure_week and week >= failure_week:
            txn = 0
            gmv = 0.0
            r_rate = prev_refund_r
            cb_rate = 0.0
            refund_amt = 0.0
            cb_amt = 0.0
            r_count = 0
            cb_count = 0
            ticket = 0.0
            is_spike = int(week in FESTIVAL_WEEKS)
            rows.append(_build_row(
                merchant_id, segment, week, gmv, txn, ticket,
                refund_amt, r_rate, cb_amt, cb_rate, r_count, cb_count,
                prev_gmv, is_spike
            ))
            prev_gmv = gmv
            continue

        # ── GMV evolution ──
        if segment == "fraudulent":
            if week <= 6:
                gmv = gmv_base * 3.0 * seasonal_multiplier(week, A, B) * festival_multiplier(week)
            else:
                gmv = gmv_base * max(0.1, 3.0 - 0.45 * (week - 6)) * seasonal_multiplier(week, A, B)
        elif segment == "growing":
            gmv = gmv_base * ((1 + growth_rate) ** week) * seasonal_multiplier(week, A, B) * festival_multiplier(week)
        elif segment == "risky":
            decay_start = 8
            if week > decay_start:
                gmv = gmv_base * ((1 - decay_rate) ** (week - decay_start))
            else:
                gmv = gmv_base
            gmv *= seasonal_multiplier(week, A, B) * festival_multiplier(week)
        else:  # healthy / stable
            noise = float(rng.normal(0, gmv_base * 0.05))
            gmv = max(100.0, gmv_base + noise) * seasonal_multiplier(week, A, B) * festival_multiplier(week)

        gmv = max(100.0, gmv)

        # ── refund rate evolution ──
        if segment == "healthy":
            refund_r = np.clip(refund_r + float(rng.normal(0, 0.005)), 0.01, 0.20)
        elif segment == "growing":
            refund_r = np.clip(refund_r - 0.002, 0.01, 0.20)
        elif segment == "risky" and week > 8:
            refund_r = np.clip(refund_r + float(rng.uniform(0.005, 0.015)), 0.01, 0.65)
        elif segment == "fraudulent" and week > failure_week - 5 if failure_week else False:
            refund_r = np.clip(refund_r + 0.05, 0.01, 0.70)
        else:
            refund_r = np.clip(refund_r + float(rng.normal(0, 0.003)), 0.01, 0.30)

        # ── chargeback rate (3-week lag on refund) ──
        refund_history.append(refund_r)
        if len(refund_history) >= 3:
            lag_refund = refund_history[-3]
            seg_cb_mult = {"healthy": 0.20, "growing": 0.15, "stable": 0.22, "risky": 0.40, "fraudulent": 0.80}
            charge_r = np.clip(lag_refund * seg_cb_mult[segment] + float(rng.normal(0, 0.003)), 0.001, 0.30)

        # ── amounts ──
        refund_amt = gmv * refund_r + float(rng.poisson(500))
        cb_amt     = gmv * charge_r + float(rng.poisson(200))
        txn        = txn_count_from_gmv(gmv, segment)
        ticket     = gmv / txn if txn > 0 else 0.0
        r_count    = max(1, int(refund_amt / (ticket + 1)))
        cb_count   = max(0, int(cb_amt / (ticket + 1)))
        is_spike   = int(week in FESTIVAL_WEEKS)

        rows.append(_build_row(
            merchant_id, segment, week, gmv, txn, ticket,
            refund_amt, refund_r, cb_amt, charge_r, r_count, cb_count,
            prev_gmv, is_spike
        ))
        prev_gmv = gmv
        prev_refund_r = refund_r

    return rows


def _build_row(mid, seg, week, gmv, txn, ticket,
               r_amt, r_rate, cb_amt, cb_rate, r_cnt, cb_cnt,
               prev_gmv, is_spike):
    gmv_pct = (gmv - prev_gmv) / (prev_gmv + 1e-6) if prev_gmv is not None else 0.0
    return {
        "merchant_id":       mid,
        "week":              week,
        "segment":           seg,
        "gmv":               round(gmv, 2),
        "txn_count":         txn,
        "avg_ticket_size":   round(ticket, 2),
        "refund_amount":     round(r_amt, 2),
        "refund_rate":       round(r_rate, 5),
        "chargeback_amount": round(cb_amt, 2),
        "chargeback_rate":   round(cb_rate, 5),
        "refund_count":      r_cnt,
        "chargeback_count":  cb_cnt,
        "gmv_prev_week":     round(prev_gmv, 2) if prev_gmv is not None else np.nan,
        "gmv_pct_change":    round(gmv_pct, 5),
        "is_seasonal_spike": is_spike,
    }


# ── rolling feature engineering ───────────────────────────────────────────────

def add_rolling_features(df):
    df = df.sort_values(["merchant_id", "week"]).reset_index(drop=True)
    grp = df.groupby("merchant_id")

    df["gmv_4w_rolling_avg"]      = grp["gmv"].transform(lambda x: x.rolling(4, min_periods=1).mean())
    df["gmv_4w_rolling_std"]      = grp["gmv"].transform(lambda x: x.rolling(4, min_periods=1).std().fillna(0))
    df["gmv_volatility"]          = df["gmv_4w_rolling_std"] / (df["gmv_4w_rolling_avg"] + 1e-6)
    df["refund_rate_4w_avg"]      = grp["refund_rate"].transform(lambda x: x.rolling(4, min_periods=1).mean())
    df["refund_acceleration"]     = grp["refund_rate"].transform(lambda x: x.diff().fillna(0))
    df["chargeback_rate_4w_avg"]  = grp["chargeback_rate"].transform(lambda x: x.rolling(4, min_periods=1).mean())
    df["chargeback_acceleration"] = grp["chargeback_rate"].transform(lambda x: x.diff().fillna(0))
    df["txn_count_volatility"]    = grp["txn_count"].transform(lambda x: x.rolling(4, min_periods=1).std().fillna(0))

    # ticket size drift (vs 4 weeks ago)
    df["ticket_size_drift"] = grp["avg_ticket_size"].transform(
        lambda x: x / x.shift(4).bfill()
    )
    df["refund_to_chargeback_ratio"] = df["refund_rate"] / (df["chargeback_rate"] + 1e-6)

    # cumulative chargeback ratio
    df["cumulative_chargeback_ratio"] = (
        grp["chargeback_amount"].transform("cumsum") /
        (grp["gmv"].transform("cumsum") + 1e-6)
    )

    # GMV trend slope (OLS over last 4 weeks) — using rolling apply
    def ols_slope(series):
        arr = series.values
        n = len(arr)
        x = np.arange(n)
        if n < 2:
            return 0.0
        slope, _, _, _, _ = stats.linregress(x, arr)
        return slope

    df["gmv_trend_slope"] = grp["gmv"].transform(
        lambda x: x.rolling(4, min_periods=2).apply(ols_slope, raw=False).fillna(0)
    )

    return df


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    os.makedirs("data/raw", exist_ok=True)

    merchants = assign_segments()

    # failure_week is assigned by inject_failures.py — placeholder NaN here
    merchants["failure_week"] = np.nan
    merchants["churned"] = 0

    # Placeholder failure for fraudulent (needed to zero out GMV in simulation)
    # Actual canonical failures come from inject_failures.py
    fraud_mask = merchants["segment"] == "fraudulent"
    merchants.loc[fraud_mask, "failure_week"] = rng.integers(10, 19, size=fraud_mask.sum())
    merchants.loc[fraud_mask, "churned"] = 1

    risky_mask = merchants["segment"] == "risky"
    fail_flags = rng.random(size=risky_mask.sum()) < 0.60
    fail_weeks = rng.integers(14, 25, size=risky_mask.sum())
    fail_weeks[~fail_flags] = 0
    merchants.loc[risky_mask, "failure_week"] = np.where(fail_flags, fail_weeks[fail_flags.cumsum() - 1] if fail_flags.any() else np.nan, np.nan)
    # simpler vectorised version:
    fw_risky = np.where(fail_flags, fail_weeks, np.nan)
    merchants.loc[risky_mask, "failure_week"] = fw_risky
    merchants.loc[risky_mask, "churned"] = fail_flags.astype(int)

    stable_mask = merchants["segment"] == "stable"
    stable_fail = rng.random(size=stable_mask.sum()) < 0.05
    merchants.loc[stable_mask, "failure_week"] = np.where(stable_fail, rng.integers(20, 25, size=stable_mask.sum()), np.nan)
    merchants.loc[stable_mask, "churned"] = stable_fail.astype(int)

    # ── simulate all merchants ──
    all_rows = []
    for _, row in merchants.iterrows():
        fw = int(row["failure_week"]) if not np.isnan(row["failure_week"]) else None
        all_rows.extend(simulate_merchant(row["merchant_id"], row["segment"], fw))

    df = pd.DataFrame(all_rows)

    # ── merge failure info ──
    df = df.merge(
        merchants[["merchant_id", "failure_week", "churned"]],
        on="merchant_id", how="left"
    )

    # ── rolling features ──
    df = add_rolling_features(df)

    # ── placeholder columns (filled by later pipeline steps) ──
    for col in ["anomaly_score", "cluster_label", "umap_x", "umap_y",
                "health_score", "survival_prob_4w", "revenue_at_risk"]:
        df[col] = np.nan

    # ── column order ──
    col_order = [
        "merchant_id", "week", "segment",
        "gmv", "txn_count", "avg_ticket_size",
        "refund_amount", "refund_rate",
        "chargeback_amount", "chargeback_rate",
        "refund_count", "chargeback_count",
        "gmv_prev_week", "gmv_pct_change",
        "gmv_4w_rolling_avg", "gmv_4w_rolling_std",
        "gmv_trend_slope", "gmv_volatility",
        "refund_rate_4w_avg", "refund_acceleration",
        "chargeback_rate_4w_avg", "chargeback_acceleration",
        "txn_count_volatility", "ticket_size_drift",
        "refund_to_chargeback_ratio", "cumulative_chargeback_ratio",
        "is_seasonal_spike",
        "failure_week", "churned",
        "anomaly_score", "cluster_label",
        "umap_x", "umap_y",
        "health_score", "survival_prob_4w", "revenue_at_risk",
    ]
    df = df[col_order]
    df.to_csv("data/raw/merchant_weekly_features.csv", index=False)
    print(f"[OK] merchant_weekly_features.csv  ->  {len(df):,} rows x {len(df.columns)} columns")
    print(df["segment"].value_counts())
    print(df[["gmv", "refund_rate", "chargeback_rate"]].describe().round(4))


if __name__ == "__main__":
    main()
