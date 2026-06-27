"""
inject_failures.py
Day 1 — Step 2
Reads:   data/raw/merchant_weekly_features.csv
Produces: data/raw/merchant_survival_table.csv (500 rows)

Applies a hazard-based failure model to produce the canonical survival table.
The failure_week / churned columns already in weekly_features were set during
simulation; this script consolidates them and computes survival-table aggregates.
"""

import numpy as np
import pandas as pd
import os

SEED = 42
rng = np.random.default_rng(SEED)


# ── hazard model (validates / re-applies failure logic) ──────────────────────

def hazard_failure_week(segment, risk_score, rng):
    """
    Return (churned: bool, failure_week: int | None).
    Mirrors the logic used in simulate_merchants.py for consistency.
    """
    if segment == "healthy":
        return False, None

    if segment == "growing":
        return False, None

    if segment == "fraudulent":
        fw = int(rng.integers(10, 19))
        return True, fw

    if segment == "risky":
        # sample failure week by evaluating sigmoid hazard each week
        for w in range(8, 25):
            p_fail = _sigmoid(risk_score * 0.30 * w)
            if rng.random() < p_fail:
                return True, w
        return False, None

    if segment == "stable":
        if rng.random() < 0.05:
            fw = int(rng.integers(20, 25))
            return True, fw
        return False, None

    return False, None


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


# ── aggregate weekly features into survival table ─────────────────────────────

def build_survival_table(weekly: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for mid, grp in weekly.groupby("merchant_id"):
        grp = grp.sort_values("week")
        segment = grp["segment"].iloc[0]

        # canonical failure info from simulated data
        churned = int(grp["churned"].iloc[0])
        fw_raw  = grp["failure_week"].iloc[0]
        failure_week = int(fw_raw) if not pd.isna(fw_raw) else None

        # duration = weeks active
        duration = failure_week if (churned and failure_week) else 24
        duration = max(1, min(duration, 24))

        # active rows only
        active = grp[grp["week"] < failure_week] if (churned and failure_week) else grp

        avg_gmv             = float(active["gmv"].mean())
        max_refund_rate     = float(active["refund_rate"].max())
        max_chargeback_rate = float(active["chargeback_rate"].max())

        # anomaly_score placeholder (filled after anomaly model runs)
        avg_anomaly_score   = np.nan

        # cluster / health score placeholders
        final_cluster       = np.nan
        final_health_score  = np.nan

        # GMV at risk = cumulative GMV in last 4 active weeks
        last4 = active.tail(4)
        total_gmv_at_risk = float(last4["gmv"].sum())

        rows.append({
            "merchant_id":         mid,
            "segment":             segment,
            "entry_week":          1,
            "duration":            duration,
            "event":               churned,
            "failure_week":        failure_week,
            "avg_gmv":             round(avg_gmv, 2),
            "max_refund_rate":     round(max_refund_rate, 5),
            "max_chargeback_rate": round(max_chargeback_rate, 5),
            "avg_anomaly_score":   avg_anomaly_score,
            "final_cluster":       final_cluster,
            "final_health_score":  final_health_score,
            "total_gmv_at_risk":   round(total_gmv_at_risk, 2),
        })

    return pd.DataFrame(rows)


# ── validation checks ─────────────────────────────────────────────────────────

def validate(survival: pd.DataFrame):
    print("\n-- Survival Table Validation ---------------------")
    print(f"Total merchants    : {len(survival)}")
    print(f"Failed (event=1)   : {survival['event'].sum()}")
    print(f"Censored (event=0) : {(survival['event'] == 0).sum()}")
    print("\nEvent rate by segment:")
    print(survival.groupby("segment")["event"].agg(["sum", "count", "mean"]).round(3))
    print("\nDuration stats:")
    print(survival["duration"].describe().round(2))
    print("\nAvg GMV by segment:")
    print(survival.groupby("segment")["avg_gmv"].mean().round(0))
    print()

    # sanity checks
    assert survival["duration"].between(1, 24).all(), "Duration out of [1,24]"
    assert survival["event"].isin([0, 1]).all(), "event must be 0 or 1"
    assert (survival.loc[survival["event"] == 1, "failure_week"].notna()).all(), \
        "All failed merchants must have a failure_week"
    assert (survival.loc[survival["event"] == 0, "failure_week"].isna()).all(), \
        "Censored merchants must have null failure_week"
    print("[OK] All assertions passed.")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs("data/raw", exist_ok=True)

    weekly_path = "data/raw/merchant_weekly_features.csv"
    if not os.path.exists(weekly_path):
        raise FileNotFoundError(
            "Run simulate_merchants.py first to generate merchant_weekly_features.csv"
        )

    weekly = pd.read_csv(weekly_path)
    print(f"Loaded {len(weekly):,} rows from merchant_weekly_features.csv")

    survival = build_survival_table(weekly)
    validate(survival)

    out_path = "data/raw/merchant_survival_table.csv"
    survival.to_csv(out_path, index=False)
    print(f"[OK] merchant_survival_table.csv  ->  {len(survival)} rows x {len(survival.columns)} columns")
    print(survival.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
