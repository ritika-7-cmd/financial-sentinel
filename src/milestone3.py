"""
Financial Sentinel — Milestone 3
==================================
Forecasting → Drift Detection → Spending Trend Alerts

This milestone moves the project from:
  "What happened?" (Milestones 1 & 2)
to:
  "What's likely to happen, and what's quietly changing?"

Three components:
  1. FORECASTING: predict next month's total spend using three methods
     (Moving Average → Exponential Smoothing → ARIMA), evaluated with
     MAE, RMSE, MAPE. Each method is a baseline for the next.

  2. DRIFT DETECTION: find categories/departments where spending has
     been slowly but consistently increasing over months — not a single
     suspicious transaction, but a structural shift nobody noticed.

  3. TREND ALERTS: produce a ranked table of "things quietly getting
     worse" — the kind of intelligence a CFO actually wants.

WHY THIS MATTERS FOR YOUR RESUME:
The forecasting section is where your statistics background (SSI)
becomes a genuine differentiator. Most ML-focused portfolios skip
time series entirely. You have ARIMA, exponential smoothing, AND
drift detection — that's a real statistics story, not just sklearn.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
import warnings, os
warnings.filterwarnings("ignore")

sns.set_theme(style="darkgrid", palette="muted")
PLOT_DIR = "outputs/plots"
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs("data/processed", exist_ok=True)

# ─────────────────────────────────────────────────────────────
# SECTION 1: LOAD DATA + BUILD MONTHLY AGGREGATES
# ─────────────────────────────────────────────────────────────
print("=" * 60)
print("SECTION 1 — LOAD + MONTHLY AGGREGATION")
print("=" * 60)

df = pd.read_csv("data/processed/transactions_m2.csv", parse_dates=["timestamp"])

# Aggregate to monthly level — forecasting works on time series,
# not individual transactions. Each data point = one month's total spend.
df["month"] = df["timestamp"].dt.to_period("M")

monthly_total = (
    df.groupby("month")["amount"]
    .sum()
    .reset_index()
    .rename(columns={"amount": "total_spend"})
)
monthly_total["month_dt"] = monthly_total["month"].dt.to_timestamp()
monthly_total = monthly_total.sort_values("month_dt").reset_index(drop=True)

# Department-level monthly spend (for drift detection later)
monthly_dept = (
    df.groupby(["month", "department"])["amount"]
    .sum()
    .reset_index()
    .rename(columns={"amount": "total_spend"})
)
monthly_dept["month_dt"] = monthly_dept["month"].dt.to_timestamp()

# Category-level monthly spend
monthly_cat = (
    df.groupby(["month", "category"])["amount"]
    .sum()
    .reset_index()
    .rename(columns={"amount": "total_spend"})
)
monthly_cat["month_dt"] = monthly_cat["month"].dt.to_timestamp()

print(f"Monthly data points: {len(monthly_total)}")
print(f"\nMonthly total spend (₹ Lakhs):")
for _, row in monthly_total.iterrows():
    bar = "█" * int(row["total_spend"] / 5_000_000)
    print(f"  {str(row['month']):<10}  ₹{row['total_spend']/100_000:>8.1f}L  {bar}")


# ─────────────────────────────────────────────────────────────
# SECTION 2: TRAIN/TEST SPLIT
# ─────────────────────────────────────────────────────────────
# We train on the first N-3 months and test on the last 3.
# This mimics real usage: you have historical data, you predict
# the next few months, and you check how close you were.
print("\n" + "=" * 60)
print("SECTION 2 — TRAIN / TEST SPLIT")
print("=" * 60)

TEST_MONTHS = 3
train = monthly_total.iloc[:-TEST_MONTHS]
test  = monthly_total.iloc[-TEST_MONTHS:]

print(f"Train: {len(train)} months ({train['month'].iloc[0]} → {train['month'].iloc[-1]})")
print(f"Test : {len(test)} months  ({test['month'].iloc[0]} → {test['month'].iloc[-1]})")

y_train = train["total_spend"].values
y_test  = test["total_spend"].values


# ─────────────────────────────────────────────────────────────
# SECTION 3: EVALUATION METRICS
# ─────────────────────────────────────────────────────────────
# MAE  = Mean Absolute Error — average rupee error per month (easy to explain)
# RMSE = Root Mean Squared Error — penalises big misses more than MAE
# MAPE = Mean Absolute Percentage Error — error as % of actual spend
#        (useful because it's scale-independent: 5% MAPE means you're
#        off by 5% regardless of whether spend is ₹10L or ₹100L)

def evaluate(actual, predicted, name):
    mae  = np.mean(np.abs(actual - predicted))
    rmse = np.sqrt(np.mean((actual - predicted) ** 2))
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    print(f"\n  {name}")
    print(f"    MAE  : ₹{mae:,.0f}  (avg rupee error per month)")
    print(f"    RMSE : ₹{rmse:,.0f}  (penalises big misses more)")
    print(f"    MAPE : {mape:.2f}%  (error as % of actual)")
    return {"model": name, "MAE": mae, "RMSE": rmse, "MAPE": mape}


# ─────────────────────────────────────────────────────────────
# SECTION 4: METHOD 1 — MOVING AVERAGE (BASELINE)
# ─────────────────────────────────────────────────────────────
# Simplest possible forecast: next month = average of last N months.
# This is the baseline every other method must beat.
# If ARIMA can't beat a 3-month moving average, ARIMA is not helping.
print("\n" + "=" * 60)
print("SECTION 4 — METHOD 1: MOVING AVERAGE (BASELINE)")
print("=" * 60)

WINDOW = 3
ma_forecast = np.full(TEST_MONTHS, y_train[-WINDOW:].mean())
results_ma = evaluate(y_test, ma_forecast, "Moving Average (3-month)")


# ─────────────────────────────────────────────────────────────
# SECTION 5: METHOD 2 — EXPONENTIAL SMOOTHING
# ─────────────────────────────────────────────────────────────
# Exponential Smoothing gives MORE weight to recent months and
# less to older ones (weights decay exponentially going back in time).
# More sophisticated than MA — recent behaviour matters more.
# Holt-Winters version adds trend and seasonality components.
print("\n" + "=" * 60)
print("SECTION 5 — METHOD 2: EXPONENTIAL SMOOTHING")
print("=" * 60)

es_model = ExponentialSmoothing(
    y_train,
    trend="add",        # model an additive trend (spend going up over time)
    seasonal=None,      # no seasonality — we don't have enough data for that
    initialization_method="estimated"
).fit(optimized=True)  # let statsmodels find optimal smoothing parameters

es_forecast = es_model.forecast(TEST_MONTHS)
results_es = evaluate(y_test, es_forecast, "Exponential Smoothing (Holt)")

print(f"\n  Smoothing level (alpha): {es_model.params['smoothing_level']:.3f}")
print(f"  (alpha close to 1 = heavily weights most recent months)")


# ─────────────────────────────────────────────────────────────
# SECTION 6: STATIONARITY CHECK (FOR ARIMA)
# ─────────────────────────────────────────────────────────────
# ARIMA requires the time series to be "stationary" — meaning its
# statistical properties (mean, variance) don't change over time.
# Financial spending data usually isn't stationary (it trends upward).
# We check with the Augmented Dickey-Fuller test:
#   p-value < 0.05 → stationary (ARIMA can use raw series)
#   p-value >= 0.05 → non-stationary → we difference the series first
print("\n" + "=" * 60)
print("SECTION 6 — STATIONARITY CHECK (ADF TEST)")
print("=" * 60)

adf_result = adfuller(y_train)
adf_pvalue = adf_result[1]
print(f"  ADF test p-value: {adf_pvalue:.4f}")
if adf_pvalue < 0.05:
    print("  → Series is stationary (p < 0.05). ARIMA can use d=0.")
    d = 0
else:
    print("  → Series is non-stationary (p >= 0.05). Will use d=1 (first difference).")
    d = 1


# ─────────────────────────────────────────────────────────────
# SECTION 7: METHOD 3 — ARIMA
# ─────────────────────────────────────────────────────────────
# ARIMA(p, d, q):
#   p = number of autoregressive terms (how many past values to use)
#   d = degree of differencing (0 if stationary, 1 if not)
#   q = number of moving average terms
# We use (1, d, 1) — simple but standard for monthly financial data.
# With only ~19 months of training data, a complex ARIMA would overfit.
print("\n" + "=" * 60)
print("SECTION 7 — METHOD 3: ARIMA")
print("=" * 60)

arima_model = ARIMA(y_train, order=(1, d, 1)).fit()
arima_forecast = arima_model.forecast(steps=TEST_MONTHS)

# 95% confidence interval for the forecast
forecast_obj = arima_model.get_forecast(steps=TEST_MONTHS)
arima_ci     = forecast_obj.conf_int(alpha=0.05)
arima_ci_arr = arima_ci.values if hasattr(arima_ci, "values") else arima_ci

results_arima = evaluate(y_test, arima_forecast, f"ARIMA(1,{d},1)")

print(f"\n  ARIMA forecast with 95% prediction interval:")
for i, (month, actual) in enumerate(zip(test["month"].values, y_test)):
    lo   = arima_ci_arr[i, 0]
    hi   = arima_ci_arr[i, 1]
    pred = float(np.array(arima_forecast)[i])
    print(f"    {month}:  Actual ₹{actual/100_000:.1f}L  |  "
          f"Predicted ₹{pred/100_000:.1f}L  |  "
          f"95% CI [₹{lo/100_000:.1f}L – ₹{hi/100_000:.1f}L]")


# ─────────────────────────────────────────────────────────────
# SECTION 8: MODEL COMPARISON
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SECTION 8 — MODEL COMPARISON")
print("=" * 60)

results_df = pd.DataFrame([results_ma, results_es, results_arima])
results_df = results_df.set_index("model")
print(results_df.round(2).to_string())

best_mape = results_df["MAPE"].idxmin()
print(f"\n  Best model by MAPE: {best_mape}")
print(f"  ({results_df.loc[best_mape, 'MAPE']:.2f}% average error on held-out months)")


# ─────────────────────────────────────────────────────────────
# SECTION 9: PLOT — FORECAST COMPARISON
# ─────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 6))

# Training data
ax.plot(train["month_dt"], y_train / 100_000,
        color="#5b7fa6", linewidth=2, label="Actual (train)", marker="o", markersize=4)

# Test actual
ax.plot(test["month_dt"], y_test / 100_000,
        color="#5b7fa6", linewidth=2, linestyle="--", marker="o", markersize=6,
        label="Actual (test)")

# Forecasts
ax.plot(test["month_dt"], ma_forecast / 100_000,
        color="#c97b7b", linewidth=1.8, marker="s", markersize=6,
        linestyle=":", label="Moving Average")
ax.plot(test["month_dt"], es_forecast / 100_000,
        color="#e8a838", linewidth=1.8, marker="^", markersize=6,
        linestyle="-.", label="Exp. Smoothing")
ax.plot(test["month_dt"], arima_forecast / 100_000,
        color="#7fb8a4", linewidth=1.8, marker="D", markersize=6,
        label=f"ARIMA(1,{d},1)")

# ARIMA confidence interval
ax.fill_between(test["month_dt"],
                arima_ci_arr[:, 0] / 100_000,
                arima_ci_arr[:, 1] / 100_000,
                color="#7fb8a4", alpha=0.15, label="ARIMA 95% CI")

ax.axvline(test["month_dt"].iloc[0], color="gray", linestyle="--",
           alpha=0.5, label="Train/test split")
ax.set_title("Monthly Spend Forecast — Three Methods Compared", fontsize=13)
ax.set_xlabel("")
ax.set_ylabel("Total Spend (₹ Lakhs)")
ax.legend(loc="upper left", fontsize=9)
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/10_forecast_comparison.png", dpi=150)
plt.close()
print(f"\n[Plot saved: {PLOT_DIR}/10_forecast_comparison.png]")


# ─────────────────────────────────────────────────────────────
# SECTION 10: DRIFT DETECTION
# ─────────────────────────────────────────────────────────────
# Drift = a slow, consistent directional change in spending.
# No single transaction looks suspicious. But the TREND is.
# This is different from anomaly detection — it's a macro signal.
#
# Method: for each department and category, fit a linear regression
# on monthly spend over time. The slope tells you:
#   - positive slope + statistically significant = spending is drifting UP
#   - we flag it if the % change over the observation window exceeds a threshold
#
# We also compute a "drift score" combining:
#   - slope magnitude (how fast is it changing?)
#   - R-squared (how consistent is the trend? noisy growth is less alarming)
#   - % total change over the period
print("\n" + "=" * 60)
print("SECTION 10 — DRIFT DETECTION")
print("=" * 60)

from scipy import stats as scipy_stats

def detect_drift(monthly_df, group_col, value_col="total_spend",
                 min_months=6, pct_change_threshold=20.0):
    """
    For each group (department or category), fit a linear trend.
    Return a ranked table of drifting groups.

    min_months: skip groups with fewer than this many data points
    pct_change_threshold: flag if fitted trend implies >X% change over window
    """
    results = []
    for group, gdf in monthly_df.groupby(group_col):
        gdf = gdf.sort_values("month_dt").reset_index(drop=True)
        if len(gdf) < min_months:
            continue

        x = np.arange(len(gdf))
        y = gdf[value_col].values

        slope, intercept, r_value, p_value, std_err = scipy_stats.linregress(x, y)

        # Fitted first and last values
        y_fitted_start = intercept
        y_fitted_end   = intercept + slope * (len(gdf) - 1)

        if y_fitted_start <= 0:
            continue

        pct_change = (y_fitted_end - y_fitted_start) / y_fitted_start * 100
        r_squared  = r_value ** 2

        # Drift score: combines magnitude, consistency, and significance
        # Only flag upward drifts (positive pct_change) that are statistically
        # significant (p < 0.10) and consistent (R² > 0.3)
        if pct_change > pct_change_threshold and p_value < 0.10 and r_squared > 0.3:
            drift_score = (abs(pct_change) / 100) * r_squared * (1 - p_value)
            results.append({
                group_col:      group,
                "pct_change":   round(pct_change, 1),
                "r_squared":    round(r_squared, 3),
                "p_value":      round(p_value, 4),
                "monthly_slope_lakhs": round(slope / 100_000, 2),
                "drift_score":  round(drift_score, 4),
                "n_months":     len(gdf),
                "spend_start_L": round(y_fitted_start / 100_000, 1),
                "spend_end_L":   round(y_fitted_end   / 100_000, 1),
            })

    out = pd.DataFrame(results)
    if out.empty:
        return out
    return out.sort_values("drift_score", ascending=False).reset_index(drop=True)


drift_dept = detect_drift(monthly_dept, "department")
drift_cat  = detect_drift(monthly_cat,  "category")

print("\n--- Department spending drift ---")
if drift_dept.empty:
    print("  No significant department drifts detected.")
else:
    print(drift_dept.to_string(index=False))

print("\n--- Category spending drift ---")
if drift_cat.empty:
    print("  No significant category drifts detected.")
else:
    print(drift_cat.to_string(index=False))


# ─────────────────────────────────────────────────────────────
# SECTION 11: TREND ALERT TABLE
# ─────────────────────────────────────────────────────────────
# Combine department and category drifts into one ranked alert table.
# This is what would appear in the dashboard as "Trend Alerts".
print("\n" + "=" * 60)
print("SECTION 11 — TREND ALERT TABLE")
print("=" * 60)

alert_rows = []
for _, row in drift_dept.iterrows():
    alert_rows.append({
        "type":    "Department",
        "name":    row["department"],
        "pct_change": row["pct_change"],
        "r_squared":  row["r_squared"],
        "monthly_slope_lakhs": row["monthly_slope_lakhs"],
        "drift_score": row["drift_score"],
        "spend_start_L": row["spend_start_L"],
        "spend_end_L":   row["spend_end_L"],
    })
for _, row in drift_cat.iterrows():
    alert_rows.append({
        "type":    "Category",
        "name":    row["category"],
        "pct_change": row["pct_change"],
        "r_squared":  row["r_squared"],
        "monthly_slope_lakhs": row["monthly_slope_lakhs"],
        "drift_score": row["drift_score"],
        "spend_start_L": row["spend_start_L"],
        "spend_end_L":   row["spend_end_L"],
    })

_ta = pd.DataFrame(alert_rows)
trend_alerts = _ta.sort_values("drift_score", ascending=False).reset_index(drop=True) if not _ta.empty else _ta

if trend_alerts.empty:
    print("  No significant drift alerts detected — spending is stable across all groups.")
else:
    print(f"\n  {'#':<4} {'Type':<12} {'Name':<22} {'Change':>8} {'R²':>6} {'Slope/mo':>10} {'Start':>8} {'End':>8}")
    print("  " + "-" * 80)
    for i, row in trend_alerts.iterrows():
        print(f"  {i+1:<4} {row['type']:<12} {row['name']:<22} "
              f"{row['pct_change']:>+7.1f}%  {row['r_squared']:>6.3f}  "
              f"₹{row['monthly_slope_lakhs']:>7.2f}L/mo  "
              f"₹{row['spend_start_L']:>6.1f}L → ₹{row['spend_end_L']:>6.1f}L")


# ─────────────────────────────────────────────────────────────
# SECTION 12: DRIFT PLOTS
# ─────────────────────────────────────────────────────────────
# Plot actual monthly spend + fitted trend line for each drifting group

def plot_drift(monthly_df, group_col, drift_df, title_prefix, filename):
    if drift_df.empty:
        return
    n = min(len(drift_df), 4)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), sharey=False)
    if n == 1:
        axes = [axes]
    for ax, (_, drow) in zip(axes, drift_df.head(n).iterrows()):
        group = drow[group_col]
        gdf = monthly_df[monthly_df[group_col] == group].sort_values("month_dt")
        x = np.arange(len(gdf))
        y = gdf["total_spend"].values
        slope, intercept, *_ = scipy_stats.linregress(x, y)
        trend = intercept + slope * x
        ax.plot(gdf["month_dt"], y / 100_000, "o-", color="#5b7fa6",
                linewidth=2, markersize=5, label="Actual")
        ax.plot(gdf["month_dt"], trend / 100_000, "--", color="#c97b7b",
                linewidth=1.8, label=f"Trend (+{drow['pct_change']:.0f}%)")
        ax.set_title(f"{group}\nR²={drow['r_squared']:.2f}, p={drow['p_value']:.3f}",
                     fontsize=10)
        ax.set_ylabel("₹ Lakhs")
        ax.tick_params(axis="x", rotation=30)
        ax.legend(fontsize=8)
    plt.suptitle(title_prefix, fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/{filename}", dpi=150)
    plt.close()
    print(f"[Plot saved: {PLOT_DIR}/{filename}]")

plot_drift(monthly_dept, "department", drift_dept,
           "Department Spending Drift", "11_dept_drift.png")
plot_drift(monthly_cat,  "category",  drift_cat,
           "Category Spending Drift",  "12_cat_drift.png")


# ─────────────────────────────────────────────────────────────
# SECTION 13: SAVE OUTPUTS
# ─────────────────────────────────────────────────────────────
monthly_total.to_csv("data/processed/monthly_total.csv", index=False)
monthly_dept.to_csv("data/processed/monthly_dept.csv",   index=False)
monthly_cat.to_csv("data/processed/monthly_cat.csv",     index=False)
results_df.to_csv("data/processed/forecast_results.csv")
if not trend_alerts.empty:
    trend_alerts.to_csv("data/processed/trend_alerts.csv", index=False)

print("\n[Saved: data/processed/monthly_total.csv]")
print("[Saved: data/processed/monthly_dept.csv]")
print("[Saved: data/processed/monthly_cat.csv]")
print("[Saved: data/processed/forecast_results.csv]")
print("[Saved: data/processed/trend_alerts.csv]")
print("\n✓ Milestone 3 complete.")
