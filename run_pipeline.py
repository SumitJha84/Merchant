"""
run_pipeline.py
Run the complete MerchantPulse pipeline from scratch.
Usage: python run_pipeline.py
"""
import subprocess, sys, time

STEPS = [
    ("Simulating merchants",       "src/data_gen/simulate_merchants.py"),
    ("Injecting failures",          "src/data_gen/inject_failures.py"),
    ("Engineering features",        "src/features/engineer_features.py"),
    ("Anomaly detection",           "src/models/anomaly.py"),
    ("Clustering (UMAP + KMeans)",  "src/models/clustering.py"),
    ("Survival analysis (CoxPH)",   "src/models/survival.py"),
    ("Health scoring (XGBoost)",    "src/models/health_score.py"),
    ("Building master dataframe",   "src/utils/master_builder.py"),
]

print("=" * 55)
print("  MerchantPulse — Full Pipeline")
print("=" * 55)

total_start = time.time()
for i, (label, script) in enumerate(STEPS, 1):
    print(f"\n[{i}/{len(STEPS)}] {label}...", end=" ", flush=True)
    t0 = time.time()
    result = subprocess.run([sys.executable, script], capture_output=True, text=True)
    elapsed = time.time() - t0
    if result.returncode == 0:
        print(f"OK ({elapsed:.1f}s)")
    else:
        print(f"FAILED ({elapsed:.1f}s)")
        print(result.stderr[-500:])
        sys.exit(1)

print("\n" + "=" * 55)
print(f"  Pipeline complete in {time.time()-total_start:.0f}s")
print("  Launch dashboard: streamlit run dashboard/app.py")
print("=" * 55)
