# MerchantPulse — Predicting Business Failure Before It Happens

[![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square)](https://python.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0-orange?style=flat-square)](https://xgboost.ai)
[![Lifelines](https://img.shields.io/badge/Survival-CoxPH-green?style=flat-square)](https://lifelines.readthedocs.io)
[![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-red?style=flat-square)](https://streamlit.io)
[![SHAP](https://img.shields.io/badge/XAI-SHAP-purple?style=flat-square)](https://shap.readthedocs.io)

> A production-grade merchant risk scoring system that identifies payment gateway merchants likely to fail — **3–4 weeks before refunds, chargebacks, and revenue losses accumulate.**

---

## The Problem

Payment gateways absorb downstream losses when merchants deteriorate. Chargebacks arrive weeks after the signal was already in transaction data. **MerchantPulse converts raw transaction behavior into a forward-looking 0–100 health score and rupee-denominated Revenue at Risk.**

---

## Key Results

| Metric | Value |
|---|---|
| C-index (Cox Survival Model) | **0.96** |
| Precision@50 (Anomaly Detection) | **1.00** |
| UMAP Silhouette Score | **0.47** |
| Portfolio Revenue at Risk modeled | **Rs 14.4 Cr** |
| Early warning lead time | **3–6 weeks before failure** |

---

## Architecture

```
Synthetic Simulator (500 merchants, 24 weeks)
        |
Feature Engineering (67 features: rolling, velocity, peer z-score, drift)
        |
   [IsolationForest]        [UMAP + KMeans k=5]
   Global + Personal         Behavioral clusters
   anomaly score             + Trajectory analysis
        |__________________________|
                   |
         Cox Proportional Hazards
         (time-varying covariates)
         P(survival > t) per merchant
                   |
         XGBoost Health Score [0-100]
         SHAP per-merchant explanation
                   |
         Revenue at Risk Engine
         Cluster Drift Detection
                   |
         Streamlit Dashboard (5 screens)
```

---

## Dataset

- **500 merchants x 24 weeks = 12,000 rows**
- Segments: Healthy (150), Growing (100), Stable (100), Risky (100), Fraudulent (50)
- Realistic: log-normal GMV, Beta refund rates, festival seasonality, 3-week chargeback lag, hazard-based failures

---

## Feature Engineering (67 features)

| Category | Key Features |
|---|---|
| Raw | GMV, txn_count, refund_rate, chargeback_rate |
| Rolling | 4w/8w mean, std, OLS trend slope |
| Personalized | Per-merchant 8w rolling z-score (refund, chargeback, GMV) |
| Peer benchmark | Within-segment z-score per week |
| Spike detection | Refund velocity spike flag (>3-sigma personal baseline) |
| Business | CB-to-refund conversion rate, settlement exposure, net GMV |

---

## Models

**Cox Proportional Hazards** (Lifelines CoxTimeVaryingFitter)
- VIF analysis applied; multicollinear features dropped
- L2 penalization (penalizer=0.1)
- C-index: 0.96 | Kaplan-Meier curves per cluster

**XGBoost Health Score**
- Target: composite of survival_prob, refund_rate, chargeback_rate, anomaly_score
- SHAP TreeExplainer for per-merchant plain-English alerts

**Anomaly Detection** (Two-layer ensemble)
- Layer 1: IsolationForest (global population)
- Layer 2: Per-merchant personalized z-score
- Ensemble: 0.55 x IF + 0.45 x personal score

**Cluster Drift Detection**
- Cosine distance between week-1 and week-24 behavioral vectors
- Failing merchants drift 12.6 UMAP units vs 7.6 for healthy merchants

---

## Revenue at Risk Formula

```
RaR = GMV_4w_avg x 4 x P(failure_4w) x (1 + chargeback_rate)
Net_RaR = RaR - (GMV_4w_avg x 4 x P(survival_4w) x 0.02)
```

---

## Dashboard (5 Screens)

| Screen | Features |
|---|---|
| Portfolio Overview | KPIs, health histogram, risk donut, cluster RaR, top-10 table, alerts |
| Cluster Explorer | UMAP scatter, color by any dimension, cluster stats |
| Merchant Drilldown | GMV trend, refund/chargeback chart, SHAP waterfall, risk card |
| Watchlist | Severity filter, drift vs health scatter, exportable table |
| What-If Simulator | Live sliders -> instant score + RaR recalculation |

---

## Installation & Running

```bash
git clone https://github.com/yourusername/merchantpulse.git
cd merchantpulse
pip install -r requirements.txt

# Run full pipeline
python src/data_gen/simulate_merchants.py
python src/data_gen/inject_failures.py
python src/features/engineer_features.py
python src/models/anomaly.py
python src/models/clustering.py
python src/models/survival.py
python src/models/health_score.py
python src/utils/master_builder.py

# Launch dashboard
streamlit run dashboard/app.py
```

---

## Project Structure

```
merchantpulse/
├── data/
│   ├── raw/            # merchant_weekly_features.csv, merchant_survival_table.csv
│   ├── processed/      # features_engineered.csv, anomaly_scores.csv, cluster_labels.csv
│   └── outputs/        # health_scores.csv, survival_predictions.csv, master_dataframe.csv
├── src/
│   ├── data_gen/       # simulate_merchants.py, inject_failures.py
│   ├── features/       # engineer_features.py
│   ├── models/         # anomaly.py, clustering.py, survival.py, health_score.py
│   └── utils/          # master_builder.py
├── dashboard/          # app.py (Streamlit)
├── models/             # .pkl artifacts
├── requirements.txt
└── README.md
```

---

## Tech Stack

Python | Pandas | NumPy | scikit-learn | XGBoost | SHAP | Lifelines | UMAP | Streamlit | Plotly

---

## Resume Bullets

```
- Built merchant failure prediction system using Cox Proportional Hazards survival
  analysis (C-index 0.96) on 12,000 synthetic fintech transaction records.

- Engineered 67 behavioral features including personalized z-score anomaly detection
  and peer-segment benchmarking; achieved Precision@50 = 1.00.

- Designed Revenue at Risk engine quantifying Rs 14.4 Cr portfolio exposure for
  executive-level rupee-denominated risk reporting.

- Deployed 5-screen Streamlit risk platform with real-time What-If simulation,
  UMAP clustering, SHAP explanations, and automated watchlist generation.
```

---

## Future Work

Real data via Kafka/Redshift | MLflow experiment tracking | Airflow weekly retraining | FastAPI scoring endpoint | Evidently AI drift monitoring | RBI-compliant audit trail
