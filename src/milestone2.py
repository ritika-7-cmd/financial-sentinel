"""
Financial Sentinel — Milestone 2
==================================
Risk Engine → Alert Prioritisation → SHAP Explanations

Reads the Milestone 1 output (transactions_m1.csv + IF model).
Produces:
  - A risk score (0-100) for every transaction
  - A ranked alert table (top-K)
  - SHAP feature importance (global + per-transaction)
  - Evaluation against true_risk labels using Precision@K / Risk-Capture@K

IMPORTANT: The risk score is validated against true_risk (not anomaly_type).
These are different labels:
  anomaly_type -> did a statistical rule fire? (used to score the detector)
  true_risk    -> should a human actually look at this? (used to score the ranker)
This split was the main fix from the original plan.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import shap, joblib, os, warnings
warnings.filterwarnings("ignore")

sns.set_theme(style="darkgrid", palette="muted")
PLOT_DIR = "outputs/plots"
os.makedirs(PLOT_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────
# SECTION 1: LOAD MILESTONE 1 OUTPUT
# ─────────────────────────────────────────────────────────────
print("=" * 60)
print("SECTION 1 — LOADING MILESTONE 1 OUTPUT")
print("=" * 60)

df  = pd.read_csv("data/processed/transactions_m1.csv", parse_dates=["timestamp"])
clf = joblib.load("models/isolation_forest.pkl")

print(f"Transactions loaded : {len(df):,}")
print(f"True-risk labels    : {df['true_risk'].sum():,}")
print(f"IF anomaly flags    : {df['if_flag'].sum():,}")


# ─────────────────────────────────────────────────────────────
# SECTION 2: RISK ENGINE
# ─────────────────────────────────────────────────────────────
# An anomaly score alone isn't enough. A ₹200 transaction that's 10 SDs
# above normal is statistically weird but financially irrelevant. A ₹5L
# transaction that's 2 SDs above normal matters more in rupees.
# Risk = f(anomaly unusualness, financial size, behavioural deviation, context)
#
# FORMULA:
#   Risk = 0.35 * A  +  0.35 * I  +  0.20 * B  +  0.10 * C
#   where all four components are normalised to [0, 1] independently.
#
# Weight justification (you should be able to say this in an interview):
#   A (anomaly score) = 0.35: core detection signal, but alone it's not
#     enough — a ₹50 duplicate matters less than a ₹5L vendor anomaly.
#   I (financial impact) = 0.35: absolute rupee exposure is what the
#     business actually cares about. Equally weighted with A.
#   B (behavioural deviation) = 0.20: how far from the employee/vendor
#     baseline? Adds personalised context.
#   C (context) = 0.10: off-hours, new vendor, weekend — weak signals
#     individually, but add up as a tiebreaker.
print("\n" + "=" * 60)
print("SECTION 2 — RISK ENGINE")
print("=" * 60)

def min_max_norm(series):
    """Normalise a series to [0, 1]. Lower if_score = more anomalous,
    so we flip it by subtracting from max."""
    rng = series.max() - series.min()
    if rng == 0:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - series.min()) / rng


# Component A: anomaly score (IF score is negative; lower = worse → flip)
# We negate so that more anomalous = higher component
df["comp_A"] = min_max_norm(-df["if_score"])

# Component I: financial impact
# Use log(amount) to avoid one ₹13L transaction dominating every other
df["comp_I"] = min_max_norm(df["log_amount"])

# Component B: behavioural deviation
# Max of employee-level and vendor-level z-scores
# Clip negative z-scores to 0 (below-average spending is not a risk)
df["comp_B"] = min_max_norm(
    np.maximum(df["emp_z_score"].clip(lower=0),
               df["vendor_z_score"].clip(lower=0))
)

# Component C: contextual risk flags
# Add up: off-hours (1) + is_weekend (0.5) + is_potential_duplicate (1.5)
df["context_score"] = (
    df["is_off_hours"] * 1.0 +
    df["is_weekend"]  * 0.5 +
    df["is_potential_duplicate"] * 1.5
)
df["comp_C"] = min_max_norm(df["context_score"])

# Final risk score: weighted sum, scaled to 0-100
W_A, W_I, W_B, W_C = 0.35, 0.35, 0.20, 0.10
df["risk_score"] = (
    W_A * df["comp_A"] +
    W_I * df["comp_I"] +
    W_B * df["comp_B"] +
    W_C * df["comp_C"]
) * 100

df["risk_score"] = df["risk_score"].round(2)

# Risk tier labels (for dashboard display)
def tier(score):
    if score >= 75: return "CRITICAL"
    if score >= 55: return "HIGH"
    if score >= 35: return "MEDIUM"
    return "LOW"

df["risk_tier"] = df["risk_score"].apply(tier)

print("\nRisk score distribution:")
print(df["risk_score"].describe().round(2))
print("\nRisk tier breakdown:")
print(df["risk_tier"].value_counts())

# PLOT 6: Risk score distribution, by true_risk label
fig, ax = plt.subplots(figsize=(11, 5))
ax.hist(df[~df["true_risk"]]["risk_score"], bins=80, alpha=0.6,
        label="Not true risk", color="#7fb8a4")
ax.hist(df[df["true_risk"]]["risk_score"],  bins=80, alpha=0.7,
        label="True risk",     color="#c97b7b")
ax.set_title("Risk Score Distribution — do high-risk transactions score higher?", fontsize=12)
ax.set_xlabel("Risk Score (0–100)")
ax.set_ylabel("Count")
ax.legend()
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/06_risk_score_dist.png", dpi=150)
plt.close()
print(f"\n[Plot saved: {PLOT_DIR}/06_risk_score_dist.png]")


# ─────────────────────────────────────────────────────────────
# SECTION 3: ALERT PRIORITISATION + EVALUATION
# ─────────────────────────────────────────────────────────────
# The business problem: "we have 900 anomalies, we can only investigate 20.
# Which 20?" — the risk score's job is to put the real ones at the top.
#
# We measure this with two metrics:
#
# Precision@K = (true-risk transactions in top K) / K
#   "If we investigate the top 20 alerts, what fraction matter?"
#
# Risk-Capture@K = (true-risk transactions in top K) / (total true-risk)
#   "Of all the real problems, how many does top-20 coverage catch?"
#
# These matter more than overall F1 for a risk-ranking system, because
# an analyst doesn't review 60,000 rows — they review the top 20.
print("\n" + "=" * 60)
print("SECTION 3 — ALERT PRIORITISATION + EVALUATION")
print("=" * 60)

df_sorted = df.sort_values("risk_score", ascending=False).reset_index(drop=True)
total_true_risk = df["true_risk"].sum()

print(f"\nTotal true-risk transactions: {total_true_risk}")
print(f"\n{'K':>6}  {'Precision@K':>12}  {'RiskCapture@K':>14}")
print("-" * 38)

k_results = []
for k in [10, 20, 50, 100, 200, 500]:
    top_k = df_sorted.head(k)
    true_risk_caught = top_k["true_risk"].sum()
    prec_at_k = true_risk_caught / k
    capture_at_k = true_risk_caught / total_true_risk
    k_results.append((k, prec_at_k, capture_at_k, true_risk_caught))
    print(f"{k:>6}  {prec_at_k:>12.3f}  {capture_at_k:>14.3f}  ({true_risk_caught} caught)")

# PLOT 7: Precision@K and RiskCapture@K curves
ks = [r[0] for r in k_results]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

ax1.plot(ks, [r[1] for r in k_results], "o-", color="#5b7fa6", linewidth=2)
ax1.set_title("Precision@K\n(fraction of top-K alerts that are true risk)", fontsize=11)
ax1.set_xlabel("K (number of alerts reviewed)")
ax1.set_ylabel("Precision@K")
ax1.set_ylim(0, 1)
ax1.set_xscale("log")

ax2.plot(ks, [r[2] for r in k_results], "o-", color="#c97b7b", linewidth=2)
ax2.set_title("Risk-Capture@K\n(fraction of all true risks caught in top K)", fontsize=11)
ax2.set_xlabel("K (number of alerts reviewed)")
ax2.set_ylabel("Risk-Capture@K")
ax2.set_ylim(0, 1)
ax2.set_xscale("log")

plt.suptitle("Alert Prioritisation Performance", fontsize=13)
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/07_alert_eval.png", dpi=150)
plt.close()
print(f"\n[Plot saved: {PLOT_DIR}/07_alert_eval.png]")

print("\n--- Top 20 Alerts (ranked by risk score) ---")
top20_cols = ["transaction_id", "timestamp", "amount", "employee_id",
              "department", "vendor_id", "category", "risk_score",
              "risk_tier", "anomaly_type", "true_risk"]
print(df_sorted[top20_cols].head(20).to_string(index=False))


# ─────────────────────────────────────────────────────────────
# SECTION 4: SHAP EXPLANATIONS
# ─────────────────────────────────────────────────────────────
# SHAP (SHapley Additive exPlanations) answers: "which features pushed
# this transaction's anomaly score up or down, and by how much?"
# It comes from game theory — Shapley values divide "credit" fairly
# among all features.
#
# KNOWN ISSUE with IF + SHAP:
# Isolation Forest's score_samples() is not a proper classification
# probability. shap.TreeExplainer works on IF but interprets the output
# as the anomaly score, not a 0/1 class. This means SHAP values are
# in "anomaly score units", which aren't intuitive. The workaround:
# we use the anomaly score directly and explain what drives it higher.
print("\n" + "=" * 60)
print("SECTION 4 — SHAP EXPLANATIONS")
print("=" * 60)

FEATURES = [
    "log_amount", "hour", "day_of_week", "is_weekend", "is_off_hours",
    "emp_z_score", "amount_vs_emp_mean",
    "vendor_z_score", "amount_vs_vendor_median",
    "amount_vs_dept_mean", "vendor_30day_count", "is_potential_duplicate",
]

X = df[FEATURES].fillna(0)

print("Computing SHAP values (this takes ~30 seconds)...")
explainer  = shap.TreeExplainer(clf)
shap_vals  = explainer.shap_values(X)

# Global importance: mean absolute SHAP value per feature
mean_abs_shap = pd.Series(np.abs(shap_vals).mean(axis=0),
                           index=FEATURES).sort_values(ascending=False)
print("\nGlobal feature importance (mean |SHAP|):")
for feat, val in mean_abs_shap.items():
    bar = "█" * int(val * 400)
    print(f"  {feat:<35} {val:.5f}  {bar}")

# PLOT 8: Global SHAP bar chart
fig, ax = plt.subplots(figsize=(10, 6))
mean_abs_shap.sort_values().plot(kind="barh", ax=ax, color="#7fb8a4")
ax.set_title("Global Feature Importance (mean |SHAP value|)", fontsize=13)
ax.set_xlabel("Mean |SHAP value|")
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/08_shap_global.png", dpi=150)
plt.close()
print(f"\n[Plot saved: {PLOT_DIR}/08_shap_global.png]")

# Per-transaction SHAP: pick the top-5 riskiest transactions and explain each
print("\n--- Per-transaction SHAP explanations (top 5 alerts) ---")
top5_idx = df_sorted.head(5).index.tolist()

for rank, idx in enumerate(top5_idx, 1):
    row = df.loc[idx]
    shap_row = pd.Series(shap_vals[idx], index=FEATURES)
    # Negate: IF score_samples is negative for anomalies, so a negative
    # SHAP value means "this feature made it MORE anomalous"
    shap_contrib = (-shap_row).sort_values(ascending=False)

    print(f"\n  Alert #{rank} | TxID: {row['transaction_id']}")
    print(f"    Amount      : ₹{row['amount']:,.0f}")
    print(f"    Risk Score  : {row['risk_score']}")
    print(f"    Anomaly type: {row['anomaly_type']}")
    print(f"    True risk   : {row['true_risk']}")
    print(f"    Why flagged:")
    for feat, contrib in shap_contrib.head(5).items():
        direction = "↑ pushes toward anomaly" if contrib > 0 else "↓ reduces anomaly signal"
        print(f"      {feat:<35} {contrib:+.5f}  {direction}")

# PLOT 9: SHAP waterfall for single most anomalous transaction
# Pick the one with the highest risk score
top1_idx = df_sorted.index[0]
shap_top1 = pd.Series(shap_vals[top1_idx], index=FEATURES)
shap_contrib_top1 = (-shap_top1).sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(10, 6))
colors = ["#c97b7b" if v > 0 else "#7fb8a4" for v in shap_contrib_top1.values]
ax.barh(shap_contrib_top1.index[::-1], shap_contrib_top1.values[::-1], color=colors[::-1])
ax.axvline(0, color="black", linewidth=0.8)
ax.set_title(f"SHAP Waterfall — Top Alert\n(TxID: {df.loc[top1_idx, 'transaction_id']}, "
             f"Risk: {df.loc[top1_idx, 'risk_score']})", fontsize=12)
ax.set_xlabel("SHAP contribution (positive = drives anomaly score higher)")
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/09_shap_waterfall.png", dpi=150)
plt.close()
print(f"\n[Plot saved: {PLOT_DIR}/09_shap_waterfall.png]")


# ─────────────────────────────────────────────────────────────
# SECTION 5: SAVE FINAL OUTPUT
# ─────────────────────────────────────────────────────────────
# Save the SHAP values alongside the transaction data
shap_df = pd.DataFrame(shap_vals, columns=[f"shap_{f}" for f in FEATURES])
df_final = pd.concat([df.reset_index(drop=True), shap_df], axis=1)
df_final.to_csv("data/processed/transactions_m2.csv", index=False)

# Save the top-100 alert table separately (this is what a dashboard would serve)
df_sorted[top20_cols + ["comp_A", "comp_I", "comp_B", "comp_C"]].head(100).to_csv("data/processed/top100_alerts.csv", index=False)

print("\n[Saved: data/processed/transactions_m2.csv]")
print("[Saved: data/processed/top100_alerts.csv]")
print("\n✓ Milestone 2 complete.")
