import os
"""
Financial Sentinel - Synthetic Data Generator
================================================
Generates a realistic company transaction dataset with:
  1. Correlated entities (employees belong to departments, vendors serve
     categories, spending patterns differ by department)
  2. Right-skewed (lognormal) amount distributions, because real money
     amounts are NEVER normally distributed - most transactions are small,
     a few are huge. If you use np.random.normal() for money, you've
     already made a beginner mistake that a reviewer will catch.
  3. TWO separate ground-truth labels per injected anomaly:
       - anomaly_type: what statistical rule was violated (for evaluating
         your anomaly detector's precision/recall)
       - true_risk: whether a human would actually care about this one
         (for evaluating your RISK SCORE - a different, harder question)
     These are deliberately NOT the same thing. A spending drift is
     statistically detectable but is not automatically "risk". A
     duplicate invoice IS risk. Conflating the two was the main design
     flaw in the original plan - fixed here.
  4. Injected anomalies are blended with graded noise (not always at
     extreme z-scores) so your detector's precision/recall numbers mean
     something, instead of trivially separating obvious outliers.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import uuid

RNG = np.random.default_rng(42)

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------
N_EMPLOYEES = 220
N_VENDORS = 65
N_TRANSACTIONS = 60_000
START_DATE = datetime(2025, 1, 1)
END_DATE = datetime(2026, 8, 1)
TOTAL_DAYS = (END_DATE - START_DATE).days

DEPARTMENTS = ["Sales", "Marketing", "HR", "Operations", "Engineering", "Finance"]
# Each department has a different "spending personality" (mean, std of
# monthly per-employee spend, in INR). This is what makes later features
# like `amount_vs_department_average` actually mean something.
DEPT_PROFILE = {
    "Sales":       {"mean": 22000, "sigma": 0.55},
    "Marketing":   {"mean": 35000, "sigma": 0.65},
    "HR":          {"mean": 9000,  "sigma": 0.40},
    "Operations":  {"mean": 15000, "sigma": 0.50},
    "Engineering": {"mean": 18000, "sigma": 0.45},
    "Finance":     {"mean": 11000, "sigma": 0.35},
}

CATEGORIES = ["Travel", "Software", "Office Supplies", "Client Entertainment",
              "Equipment", "Consulting", "Utilities"]
PAYMENT_METHODS = ["Corporate Card", "Bank Transfer", "Reimbursement"]
LOCATIONS = ["Mumbai", "Pune", "Bangalore", "Delhi", "Chennai", "Remote"]


# ----------------------------------------------------------------------
# STEP 1: EMPLOYEES
# ----------------------------------------------------------------------
def generate_employees(n=N_EMPLOYEES):
    depts = RNG.choice(DEPARTMENTS, size=n)
    rows = []
    for i, dept in enumerate(depts):
        profile = DEPT_PROFILE[dept]
        # lognormal mean is set so that np.exp(mu) ~= profile mean.
        # This is the standard trick for generating right-skewed data
        # with a controllable "typical" value.
        mu = np.log(profile["mean"])
        sigma = profile["sigma"]
        base_monthly_spend = RNG.lognormal(mu, sigma)
        rows.append({
            "employee_id": f"EMP{i:04d}",
            "department": dept,
            "base_monthly_spend": round(base_monthly_spend, 2),
            "join_date": (START_DATE + timedelta(days=int(RNG.integers(0, 400)))).date(),
        })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# STEP 2: VENDORS
# ----------------------------------------------------------------------
def generate_vendors(n=N_VENDORS):
    rows = []
    for i in range(n):
        category = RNG.choice(CATEGORIES)
        # Vendor "typical size" varies a lot - a software vendor billing
        # ₹50k/month and a stationery vendor billing ₹2k/month should NOT
        # be judged against the same absolute threshold. This is why
        # vendor-relative features matter later.
        typical_mean = RNG.lognormal(np.log(12000), 0.9)
        rows.append({
            "vendor_id": f"VEN{i:03d}",
            "category": category,
            "typical_txn_mean": round(typical_mean, 2),
            "typical_txn_sigma": round(RNG.uniform(0.25, 0.55), 2),
            "is_new_vendor": RNG.random() < 0.08,
        })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# STEP 3: NORMAL (NON-ANOMALOUS) TRANSACTIONS
# ----------------------------------------------------------------------
def random_business_timestamp():
    """Most transactions happen 9am-8pm on weekdays - not uniformly
    across all 168 hours of the week. We bias sampling accordingly."""
    day_offset = int(RNG.integers(0, TOTAL_DAYS))
    date = START_DATE + timedelta(days=day_offset)
    # Skew toward weekdays: if weekend, 80% chance we resample to Monday
    if date.weekday() >= 5 and RNG.random() < 0.8:
        date = date - timedelta(days=date.weekday() - 4)
    hour = int(np.clip(RNG.normal(14, 3), 8, 20))
    minute = int(RNG.integers(0, 60))
    return date.replace(hour=hour, minute=minute)


def generate_normal_transactions(employees, vendors, n):
    rows = []
    emp_ids = employees["employee_id"].values
    for _ in range(n):
        emp = employees[employees["employee_id"] ==
                         RNG.choice(emp_ids)].iloc[0]
        vendor = vendors[vendors["category"] ==
                          RNG.choice(CATEGORIES)]
        if vendor.empty:
            vendor = vendors.sample(1, random_state=int(RNG.integers(0, 1e6)))
        vendor = vendor.sample(1, random_state=int(RNG.integers(0, 1e6))).iloc[0]

        amount = RNG.lognormal(np.log(vendor["typical_txn_mean"]),
                                vendor["typical_txn_sigma"])
        ts = random_business_timestamp()

        rows.append({
            "transaction_id": f"TX{uuid.uuid4().hex[:8].upper()}",
            "timestamp": ts,
            "amount": round(float(amount), 2),
            "employee_id": emp["employee_id"],
            "department": emp["department"],
            "vendor_id": vendor["vendor_id"],
            "category": vendor["category"],
            "payment_method": RNG.choice(PAYMENT_METHODS, p=[0.6, 0.3, 0.1]),
            "location": RNG.choice(LOCATIONS),
            "invoice_id": f"INV{uuid.uuid4().hex[:6].upper()}",
            "anomaly_type": "none",
            "true_risk": False,
        })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# STEP 4: INJECTED ANOMALIES (graded severity, dual-labeled)
# ----------------------------------------------------------------------
def inject_anomalies(df, employees, vendors, n_each=120):
    """
    Each anomaly type gets a RANGE of severities (mild -> extreme),
    not just extreme cases. This matters: a detector that only catches
    obvious 10-sigma outliers looks great on paper and is useless in
    production, where most real anomalies are borderline.
    true_risk is set independently of anomaly_type - this is the fix
    for the risk-score validation problem.
    """
    injected = []

    # A. Amount anomaly (graded 3x-15x normal) -> genuinely risky
    for _ in range(n_each):
        base = df.sample(1, random_state=int(RNG.integers(0, 1e6))).iloc[0].copy()
        multiplier = RNG.uniform(3, 15)
        base["amount"] = round(base["amount"] * multiplier, 2)
        base["transaction_id"] = f"TX{uuid.uuid4().hex[:8].upper()}"
        base["anomaly_type"] = "amount_anomaly"
        base["true_risk"] = bool(multiplier > 5)  # mild 3x bumps aren't automatically "risk"
        injected.append(base)

    # B. Employee behavior anomaly -> risky
    for _ in range(n_each):
        emp = employees.sample(1, random_state=int(RNG.integers(0, 1e6))).iloc[0]
        base = df.sample(1, random_state=int(RNG.integers(0, 1e6))).iloc[0].copy()
        base["employee_id"] = emp["employee_id"]
        base["department"] = emp["department"]
        base["amount"] = round(emp["base_monthly_spend"] * RNG.uniform(2.5, 6), 2)
        base["transaction_id"] = f"TX{uuid.uuid4().hex[:8].upper()}"
        base["anomaly_type"] = "employee_behavior_anomaly"
        base["true_risk"] = True
        injected.append(base)

    # C. Vendor anomaly -> risky
    for _ in range(n_each):
        vendor = vendors.sample(1, random_state=int(RNG.integers(0, 1e6))).iloc[0]
        base = df.sample(1, random_state=int(RNG.integers(0, 1e6))).iloc[0].copy()
        base["vendor_id"] = vendor["vendor_id"]
        base["category"] = vendor["category"]
        base["amount"] = round(vendor["typical_txn_mean"] * RNG.uniform(4, 12), 2)
        base["transaction_id"] = f"TX{uuid.uuid4().hex[:8].upper()}"
        base["anomaly_type"] = "vendor_anomaly"
        base["true_risk"] = True
        injected.append(base)

    # D. Duplicate transaction -> risky (near-certain real-world problem)
    for _ in range(n_each // 2):
        base = df.sample(1, random_state=int(RNG.integers(0, 1e6))).iloc[0].copy()
        dup = base.copy()
        dup["transaction_id"] = f"TX{uuid.uuid4().hex[:8].upper()}"
        dup["timestamp"] = base["timestamp"] + timedelta(minutes=int(RNG.integers(1, 90)))
        dup["anomaly_type"] = "duplicate_transaction"
        dup["true_risk"] = True
        injected.append(dup)

    # E. Time anomaly (odd hour) -> NOT automatically risky on its own
    for _ in range(n_each):
        base = df.sample(1, random_state=int(RNG.integers(0, 1e6))).iloc[0].copy()
        odd_hour = int(RNG.choice([0, 1, 2, 3, 4, 23]))
        base["timestamp"] = base["timestamp"].replace(hour=odd_hour)
        base["transaction_id"] = f"TX{uuid.uuid4().hex[:8].upper()}"
        base["anomaly_type"] = "time_anomaly"
        base["true_risk"] = False  # flag as worth a statistical look, not inherently risk
        injected.append(base)

    # F. Frequency anomaly (vendor billed unusually often) -> risky
    for _ in range(n_each // 3):
        vendor = vendors.sample(1, random_state=int(RNG.integers(0, 1e6))).iloc[0]
        burst_start = START_DATE + timedelta(days=int(RNG.integers(0, TOTAL_DAYS - 5)))
        for k in range(RNG.integers(6, 12)):
            base = df.sample(1, random_state=int(RNG.integers(0, 1e6))).iloc[0].copy()
            base["vendor_id"] = vendor["vendor_id"]
            base["category"] = vendor["category"]
            base["timestamp"] = burst_start + timedelta(hours=int(RNG.integers(0, 72)))
            base["transaction_id"] = f"TX{uuid.uuid4().hex[:8].upper()}"
            base["anomaly_type"] = "frequency_anomaly"
            base["true_risk"] = True
            injected.append(base)

    return pd.DataFrame(injected)


def main():
    print("Generating employees...")
    employees = generate_employees()
    print("Generating vendors...")
    vendors = generate_vendors()
    print(f"Generating {N_TRANSACTIONS} normal transactions...")
    normal_txns = generate_normal_transactions(employees, vendors, N_TRANSACTIONS)
    print("Injecting labeled anomalies...")
    anomalies = inject_anomalies(normal_txns, employees, vendors)

    full = pd.concat([normal_txns, anomalies], ignore_index=True)
    full = full.sort_values("timestamp").reset_index(drop=True)

    out_dir = "data/raw"
    os.makedirs(out_dir, exist_ok=True)
    employees.to_csv(f"{out_dir}/employees.csv", index=False)
    vendors.to_csv(f"{out_dir}/vendors.csv", index=False)
    full.to_csv(f"{out_dir}/transactions.csv", index=False)

    print("\n--- Summary ---")
    print(f"Total transactions: {len(full)}")
    print(full["anomaly_type"].value_counts())
    print(f"\ntrue_risk=True count: {full['true_risk'].sum()}")
    print(f"\nAmount stats:\n{full['amount'].describe()}")


if __name__ == "__main__":
    main()
