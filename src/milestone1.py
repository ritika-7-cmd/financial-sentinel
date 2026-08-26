"""
Financial Sentinel — Milestone 1
==================================
EDA → Feature Engineering → Statistical Baseline → Isolation Forest

Run this file as a script. It will:
  1. Load the raw data
  2. Run EDA and print key statistics
  3. Engineer all features
  4. Run z-score / IQR baseline detector
  5. Train Isolation Forest
  6. Compare baseline vs Isolation Forest on precision / recall / F1
  7. Save the enriched feature dataset for Milestone 2

Every section is commented to explain WHY, not just WHAT.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")          # no GUI needed — saves plots as PNG files
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
from sklearn.preprocessing import LabelEncoder
import warnings, os
warnings.filterwarnings("ignore")

sns.set_theme(style="darkgrid", palette="muted")
PLOT_DIR = "outputs/plots"
os.makedirs(PLOT_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────
# SECTION 1: LOAD DATA
# ─────────────────────────────────────────────────────────────
print("=" * 60)
print("SECTION 1 — LOADING DATA")
print("=" * 60)

df = pd.read_csv("data/raw/transactions.csv", parse_dates=["timestamp"])
employees = pd.read_csv("data/raw/employees.csv")
vendors   = pd.read_csv("data/raw/vendors.csv")

print(f"Transactions : {len(df):,}")
print(f"Employees    : {len(employees):,}")
print(f"Vendors      : {len(vendors):,}")
print(f"\nDate range   : {df['timestamp'].min().date()} → {df['timestamp'].max().date()}")
print(f"Anomalies    : {(df['anomaly_type'] != 'none').sum():,} injected")
print(f"True-risk txn: {df['true_risk'].sum():,}")


# ─────────────────────────────────────────────────────────────
# SECTION 2: EDA
# ─────────────────────────────────────────────────────────────
# EDA (Exploratory Data Analysis) means: look at the data before
# you model anything. The goal is to understand distributions,
# spot obvious patterns, and avoid making modelling decisions
# based on wrong assumptions.
print("\n" + "=" * 60)
print("SECTION 2 — EDA")
print("=" * 60)

print("\n--- Amount distribution ---")
desc = df["amount"].describe(percentiles=[.25, .5, .75, .90, .95, .99])
print(desc.round(2))

# WHY THIS MATTERS: mean >> median means the data is right-skewed
# (a few very large transactions pull the mean up). This tells you:
#   - Don't use mean as "normal" — it's inflated by big outliers
#   - Log-transform will be useful for modelling later
print(f"\nMean / Median ratio: {df['amount'].mean() / df['amount'].median():.2f}x")
print("(>1.5 = strongly right-skewed; expected for money data)")

print("\n--- Spend by department ---")
dept_spend = df.groupby("department")["amount"].agg(["mean", "median", "sum", "count"])
dept_spend["sum_lakhs"] = (dept_spend["sum"] / 100_000).round(1)
print(dept_spend.round(0))

print("\n--- Top 10 vendors by total spend ---")
vendor_spend = df.groupby("vendor_id")["amount"].sum().sort_values(ascending=False).head(10)
print(vendor_spend.round(0))

print("\n--- Transactions by hour of day ---")
df["hour"] = df["timestamp"].dt.hour
hour_counts = df.groupby("hour").size()
print(hour_counts)

print("\n--- Payment method split ---")
print(df["payment_method"].value_counts(normalize=True).round(3))

# PLOT 1: Amount distribution (raw vs log-scale)
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].hist(df["amount"], bins=80, color="#5b7fa6", edgecolor="white", linewidth=0.3)
axes[0].set_title("Transaction Amounts (raw)", fontsize=13)
axes[0].set_xlabel("Amount (₹)")
axes[0].set_ylabel("Count")

axes[1].hist(np.log1p(df["amount"]), bins=80, color="#7fb8a4", edgecolor="white", linewidth=0.3)
axes[1].set_title("Transaction Amounts (log scale)", fontsize=13)
axes[1].set_xlabel("log(1 + Amount)")
axes[1].set_ylabel("Count")

plt.suptitle("Right-skewed distribution — log transform needed for modelling", fontsize=11, style="italic")
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/01_amount_distribution.png", dpi=150)
plt.close()
print(f"\n[Plot saved: {PLOT_DIR}/01_amount_distribution.png]")

# PLOT 2: Spend by department
fig, ax = plt.subplots(figsize=(10, 5))
dept_med = df.groupby("department")["amount"].median().sort_values(ascending=False)
dept_med.plot(kind="bar", ax=ax, color="#7fb8a4", edgecolor="white")
ax.set_title("Median Transaction Amount by Department", fontsize=13)
ax.set_xlabel("")
ax.set_ylabel("Median Amount (₹)")
ax.tick_params(axis="x", rotation=30)
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/02_spend_by_dept.png", dpi=150)
plt.close()
print(f"[Plot saved: {PLOT_DIR}/02_spend_by_dept.png]")

# PLOT 3: Transactions by hour
fig, ax = plt.subplots(figsize=(11, 4))
hour_counts.plot(kind="bar", ax=ax, color="#a57fb8", edgecolor="white")
ax.set_title("Transaction Frequency by Hour of Day", fontsize=13)
ax.set_xlabel("Hour")
ax.set_ylabel("Count")
ax.axvspan(0, 7.5, alpha=0.08, color="red", label="Off-hours (risk signal)")
ax.axvspan(20.5, 23, alpha=0.08, color="red")
ax.legend()
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/03_txn_by_hour.png", dpi=150)
plt.close()
print(f"[Plot saved: {PLOT_DIR}/03_txn_by_hour.png]")

# PLOT 4: Monthly total spend trend
df["month"] = df["timestamp"].dt.to_period("M")
monthly = df.groupby("month")["amount"].sum() / 100_000
fig, ax = plt.subplots(figsize=(13, 4))
monthly.plot(ax=ax, color="#5b7fa6", linewidth=2, marker="o", markersize=4)
ax.set_title("Total Monthly Spend (₹ Lakhs)", fontsize=13)
ax.set_xlabel("")
ax.set_ylabel("₹ Lakhs")
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/04_monthly_trend.png", dpi=150)
plt.close()
print(f"[Plot saved: {PLOT_DIR}/04_monthly_trend.png]")


# ─────────────────────────────────────────────────────────────
# SECTION 3: FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────
# Raw columns (amount, timestamp) are not enough. A ₹50,000 transaction
# is normal for a Marketing vendor, suspicious for a stationery supplier.
# We need CONTEXT — that's what features encode.
print("\n" + "=" * 60)
print("SECTION 3 — FEATURE ENGINEERING")
print("=" * 60)

# --- Time features (from timestamp) ---
df["hour"]       = df["timestamp"].dt.hour
df["day_of_week"]= df["timestamp"].dt.dayofweek   # 0=Mon, 6=Sun
df["month_num"]  = df["timestamp"].dt.month
df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
df["is_off_hours"]= ((df["hour"] < 8) | (df["hour"] > 20)).astype(int)

# --- Employee-level aggregates ---
# For each employee, compute their "normal" behaviour across ALL their
# historical transactions. Then compare each individual transaction
# against that personal baseline.
emp_stats = df[df["anomaly_type"] == "none"].groupby("employee_id")["amount"].agg(
    emp_mean="mean",
    emp_median="median",
    emp_std="std",
    emp_txn_count="count"
).reset_index()
emp_stats["emp_std"] = emp_stats["emp_std"].fillna(1)   # single-txn employees: std=0 → 1 avoids div/0

df = df.merge(emp_stats, on="employee_id", how="left")
df["emp_mean"]      = df["emp_mean"].fillna(df["amount"].median())
df["emp_std"]       = df["emp_std"].fillna(df["amount"].std())
df["emp_txn_count"] = df["emp_txn_count"].fillna(1)

# Employee deviation: how many "personal standard deviations" above normal?
# A ₹90k spend from someone who always spends ₹20k is 3.5 personal-SDs above.
# Same ₹90k from a high-roller who normally spends ₹80k is barely 1 SD above.
df["amount_vs_emp_mean"]    = df["amount"] / df["emp_mean"]
df["emp_z_score"]           = (df["amount"] - df["emp_mean"]) / df["emp_std"]

# --- Vendor-level aggregates ---
vendor_stats = df[df["anomaly_type"] == "none"].groupby("vendor_id")["amount"].agg(
    vendor_mean="mean",
    vendor_median="median",
    vendor_std="std",
    vendor_txn_count="count"
).reset_index()
vendor_stats["vendor_std"] = vendor_stats["vendor_std"].fillna(1)

df = df.merge(vendor_stats, on="vendor_id", how="left")
df["vendor_mean"]      = df["vendor_mean"].fillna(df["amount"].median())
df["vendor_std"]       = df["vendor_std"].fillna(df["amount"].std())
df["vendor_txn_count"] = df["vendor_txn_count"].fillna(1)

df["amount_vs_vendor_median"] = df["amount"] / df["vendor_median"]
df["vendor_z_score"]          = (df["amount"] - df["vendor_mean"]) / df["vendor_std"]

# --- Department-level aggregates ---
dept_stats = df[df["anomaly_type"] == "none"].groupby("department")["amount"].agg(
    dept_mean="mean",
    dept_std="std"
).reset_index()
df = df.merge(dept_stats, on="department", how="left")
df["amount_vs_dept_mean"] = df["amount"] / df["dept_mean"]

# --- Vendor frequency: transactions per vendor per rolling 30-day window ---
# This catches the "vendor billed 11 times in 3 days" anomaly.
df = df.sort_values("timestamp").reset_index(drop=True)
df["vendor_30day_count"] = (
    df.groupby("vendor_id")["timestamp"]
    .transform(lambda ts: ts.expanding().count())
)

# --- Duplicate detection feature ---
# Flag if same (invoice_id, vendor_id, amount) appeared before
df["dup_key"] = df["invoice_id"].astype(str) + "_" + df["vendor_id"] + "_" + df["amount"].astype(str)
df["is_potential_duplicate"] = df.duplicated(subset=["dup_key"], keep="first").astype(int)

# --- Log amount (for models — raw amount is too skewed) ---
df["log_amount"] = np.log1p(df["amount"])

print("Features created:")
new_features = [
    "hour", "day_of_week", "is_weekend", "is_off_hours",
    "emp_mean", "emp_std", "emp_txn_count",
    "amount_vs_emp_mean", "emp_z_score",
    "vendor_mean", "vendor_std", "vendor_txn_count",
    "amount_vs_vendor_median", "vendor_z_score",
    "amount_vs_dept_mean", "vendor_30day_count",
    "is_potential_duplicate", "log_amount"
]
for f in new_features:
    print(f"  + {f}")


# ─────────────────────────────────────────────────────────────
# SECTION 4: STATISTICAL BASELINE DETECTOR
# ─────────────────────────────────────────────────────────────
# Before using ML, build a rule-based baseline. This serves two purposes:
#   1. It's a benchmark — ML should beat this, not just match it
#   2. It's interpretable — you can explain to anyone why a transaction
#      was flagged without needing to understand Isolation Forest
print("\n" + "=" * 60)
print("SECTION 4 — STATISTICAL BASELINE DETECTOR")
print("=" * 60)

# A transaction is "suspicious" by the baseline if ANY of these fire:
#   (a) Its emp_z_score > 3.5  (way above employee personal average)
#   (b) Its vendor_z_score > 3.5
#   (c) It's off-hours
#   (d) It's a potential duplicate
#   (e) High vendor frequency (> 8 txns in expanding window)

df["stat_flag"] = (
    (df["emp_z_score"] > 3.5) |
    (df["vendor_z_score"] > 3.5) |
    (df["is_off_hours"] == 1) |
    (df["is_potential_duplicate"] == 1) |
    (df["vendor_30day_count"] > 8)
).astype(int)

y_true = (df["anomaly_type"] != "none").astype(int)
stat_prec = precision_score(y_true, df["stat_flag"])
stat_rec  = recall_score(y_true, df["stat_flag"])
stat_f1   = f1_score(y_true, df["stat_flag"])

print(f"Statistical baseline flags: {df['stat_flag'].sum():,} transactions")
print(f"  Precision : {stat_prec:.3f}  (of flagged, how many are real anomalies?)")
print(f"  Recall    : {stat_rec:.3f}  (of real anomalies, how many did we catch?)")
print(f"  F1        : {stat_f1:.3f}  (harmonic mean of above two)")

# What do these numbers mean?
# Precision = 0.30 means 70% of your flags are false alarms (annoying)
# Recall = 0.60 means you caught 60% of real anomalies (missed 40%)
# F1 balances both — higher is always better


# ─────────────────────────────────────────────────────────────
# SECTION 5: ISOLATION FOREST
# ─────────────────────────────────────────────────────────────
# Isolation Forest is an ML model specifically designed for anomaly
# detection. The idea is simple: anomalies are "isolated" quickly in
# a decision tree — they need fewer random cuts to separate them from
# the rest of the data. Normal points take many more cuts.
# It outputs an anomaly_score — lower = more anomalous.
# contamination=0.015 tells it "expect ~1.5% of data to be anomalous."
print("\n" + "=" * 60)
print("SECTION 5 — ISOLATION FOREST")
print("=" * 60)

FEATURES = [
    "log_amount",
    "hour", "day_of_week", "is_weekend", "is_off_hours",
    "emp_z_score", "amount_vs_emp_mean",
    "vendor_z_score", "amount_vs_vendor_median",
    "amount_vs_dept_mean",
    "vendor_30day_count",
    "is_potential_duplicate",
]

X = df[FEATURES].fillna(0)

# Contamination = expected anomaly rate in data
# We have 872 anomalies in 60,872 rows = 0.0143
contamination = (df["anomaly_type"] != "none").mean()
print(f"Contamination rate: {contamination:.4f}")

clf = IsolationForest(
    n_estimators=200,      # number of trees — more = more stable
    contamination=contamination,
    random_state=42,
    n_jobs=-1              # use all CPU cores
)
clf.fit(X)

# predict() gives +1 (normal) or -1 (anomaly)
# score_samples() gives the raw anomaly score (-1 = most anomalous)
df["if_pred"]  = clf.predict(X)
df["if_score"] = clf.score_samples(X)   # lower = more anomalous
df["if_flag"]  = (df["if_pred"] == -1).astype(int)

if_prec = precision_score(y_true, df["if_flag"])
if_rec  = recall_score(y_true, df["if_flag"])
if_f1   = f1_score(y_true, df["if_flag"])

print(f"\nIsolation Forest flags: {df['if_flag'].sum():,} transactions")
print(f"  Precision : {if_prec:.3f}")
print(f"  Recall    : {if_rec:.3f}")
print(f"  F1        : {if_f1:.3f}")

print("\n--- Comparison ---")
print(f"{'Metric':<12} {'Baseline':>10} {'IF':>10} {'Winner':>10}")
print("-" * 45)
for name, b, i in [("Precision", stat_prec, if_prec),
                    ("Recall",    stat_rec,  if_rec),
                    ("F1",        stat_f1,   if_f1)]:
    winner = "IF" if i > b else "Baseline" if b > i else "Tie"
    print(f"{name:<12} {b:>10.3f} {i:>10.3f} {winner:>10}")

print("\n--- Anomaly type breakdown (IF detections) ---")
detected = df[df["if_flag"] == 1]["anomaly_type"].value_counts()
total    = df["anomaly_type"].value_counts()
for at in total.index:
    det = detected.get(at, 0)
    tot = total[at]
    print(f"  {at:<35}: caught {det}/{tot}  ({100*det/tot:.0f}%)")

# PLOT 5: Anomaly score distribution
fig, ax = plt.subplots(figsize=(11, 5))
ax.hist(df[y_true == 0]["if_score"], bins=80, alpha=0.6, label="Normal", color="#7fb8a4")
ax.hist(df[y_true == 1]["if_score"], bins=80, alpha=0.7, label="Anomaly", color="#c97b7b")
ax.set_title("Isolation Forest Anomaly Score Distribution", fontsize=13)
ax.set_xlabel("Anomaly Score (lower = more anomalous)")
ax.set_ylabel("Count")
ax.legend()
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/05_if_score_dist.png", dpi=150)
plt.close()
print(f"\n[Plot saved: {PLOT_DIR}/05_if_score_dist.png]")

# SAVE ENRICHED DATASET FOR MILESTONE 2
os.makedirs("data/processed", exist_ok=True)
df.to_csv("data/processed/transactions_m1.csv", index=False)
import joblib
os.makedirs("models", exist_ok=True)
joblib.dump(clf, "models/isolation_forest.pkl")
print("\n[Saved: data/processed/transactions_m1.csv]")
print("[Saved: models/isolation_forest.pkl]")

print("\n✓ Milestone 1 complete.")
