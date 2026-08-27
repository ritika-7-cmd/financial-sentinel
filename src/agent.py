"""
Financial Sentinel — AI Agent
================================
A tool-calling AI analyst that answers questions about financial risk
by calling real data functions — not by making things up.

HOW IT WORKS:
  1. User asks a question in plain English
  2. The agent decides which tool(s) to call
  3. Tools fetch real numbers from your processed data
  4. The agent synthesises an answer grounded in that evidence

WHY THIS DESIGN MATTERS:
  The LLM has NO direct access to your data files. It can only see
  what the tools return. This means:
    - Every claim in the answer is traceable to a real number
    - The agent cannot hallucinate financial figures
    - You can audit every response by checking tool call logs

TOOLS AVAILABLE:
  get_top_risks(k)              — top-K highest risk score transactions
  get_transaction(tx_id)        — full details + SHAP for one transaction
  get_vendor_history(vendor_id) — all transactions for a vendor
  get_employee_history(emp_id)  — all transactions for an employee
  get_department_summary(dept)  — spending summary for a department
  get_forecast_results()        — model comparison table
  get_risk_summary()            — overall portfolio risk statistics
  get_trend_alerts()            — drift detection results
"""

import os, json
import pandas as pd
import numpy as np
from google import genai
from google.genai import types

# ─────────────────────────────────────────────────────────────
# DATA LOADING (done once at import time)
# ─────────────────────────────────────────────────────────────
_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PROC = os.path.join(_BASE, "data", "processed")

df_txn      = pd.read_csv(os.path.join(_PROC, "transactions_m2.csv"), parse_dates=["timestamp"])
df_alerts   = pd.read_csv(os.path.join(_PROC, "top100_alerts.csv"))
df_forecast = pd.read_csv(os.path.join(_PROC, "forecast_results.csv"))
df_monthly  = pd.read_csv(os.path.join(_PROC, "monthly_total.csv"))
df_dept     = pd.read_csv(os.path.join(_PROC, "monthly_dept.csv"))

_trend_path = os.path.join(_PROC, "trend_alerts.csv")
df_trends   = pd.read_csv(_trend_path) if os.path.exists(_trend_path) else pd.DataFrame()

# SHAP columns (used in get_transaction)
SHAP_COLS = [c for c in df_txn.columns if c.startswith("shap_")]
FEATURES  = [c.replace("shap_", "") for c in SHAP_COLS]


# ─────────────────────────────────────────────────────────────
# TOOL FUNCTIONS — these return plain dicts (JSON-serialisable)
# ─────────────────────────────────────────────────────────────

def get_top_risks(k: int = 10) -> dict:
    """Return the top-K transactions by risk score."""
    k = min(int(k), 50)
    top = df_txn.sort_values("risk_score", ascending=False).head(k)
    rows = []
    for _, r in top.iterrows():
        rows.append({
            "transaction_id": r["transaction_id"],
            "timestamp":      str(r["timestamp"]),
            "amount_inr":     round(float(r["amount"]), 2),
            "employee_id":    r["employee_id"],
            "department":     r["department"],
            "vendor_id":      r["vendor_id"],
            "category":       r["category"],
            "risk_score":     round(float(r["risk_score"]), 2),
            "risk_tier":      r["risk_tier"],
            "anomaly_type":   r["anomaly_type"],
            "true_risk":      bool(r["true_risk"]),
        })
    return {"top_risks": rows, "total_returned": len(rows)}


def get_transaction(transaction_id: str) -> dict:
    """Return full details + SHAP explanation for one transaction."""
    row = df_txn[df_txn["transaction_id"] == transaction_id]
    if row.empty:
        return {"error": f"Transaction {transaction_id} not found."}
    r = row.iloc[0]

    # SHAP: top 5 drivers, negated because IF score is inverted
    shap_vals = {f: float(-r[f"shap_{f}"]) for f in FEATURES if f"shap_{f}" in r.index}
    top_shap  = sorted(shap_vals.items(), key=lambda x: abs(x[1]), reverse=True)[:5]

    return {
        "transaction_id": r["transaction_id"],
        "timestamp":      str(r["timestamp"]),
        "amount_inr":     round(float(r["amount"]), 2),
        "employee_id":    r["employee_id"],
        "department":     r["department"],
        "vendor_id":      r["vendor_id"],
        "category":       r["category"],
        "payment_method": r["payment_method"],
        "risk_score":     round(float(r["risk_score"]), 2),
        "risk_tier":      r["risk_tier"],
        "anomaly_type":   r["anomaly_type"],
        "true_risk":      bool(r["true_risk"]),
        "employee_avg_spend": round(float(r.get("emp_mean", 0)), 2),
        "vendor_avg_spend":   round(float(r.get("vendor_mean", 0)), 2),
        "emp_z_score":        round(float(r.get("emp_z_score", 0)), 2),
        "vendor_z_score":     round(float(r.get("vendor_z_score", 0)), 2),
        "shap_top_drivers": [{"feature": f, "contribution": round(v, 5)} for f, v in top_shap],
    }


def get_vendor_history(vendor_id: str) -> dict:
    """Return summary statistics and recent transactions for a vendor."""
    vdf = df_txn[df_txn["vendor_id"] == vendor_id].sort_values("timestamp")
    if vdf.empty:
        return {"error": f"Vendor {vendor_id} not found."}
    return {
        "vendor_id":       vendor_id,
        "category":        vdf["category"].iloc[0],
        "total_transactions": len(vdf),
        "total_spend_inr": round(float(vdf["amount"].sum()), 2),
        "avg_transaction": round(float(vdf["amount"].mean()), 2),
        "median_transaction": round(float(vdf["amount"].median()), 2),
        "max_transaction": round(float(vdf["amount"].max()), 2),
        "flagged_transactions": int((vdf["anomaly_type"] != "none").sum()),
        "high_risk_transactions": int(vdf["true_risk"].sum()),
        "recent_5": [
            {
                "transaction_id": r["transaction_id"],
                "timestamp": str(r["timestamp"]),
                "amount_inr": round(float(r["amount"]), 2),
                "risk_score": round(float(r["risk_score"]), 2),
                "anomaly_type": r["anomaly_type"],
            }
            for _, r in vdf.tail(5).iterrows()
        ],
    }


def get_employee_history(employee_id: str) -> dict:
    """Return summary statistics and recent transactions for an employee."""
    edf = df_txn[df_txn["employee_id"] == employee_id].sort_values("timestamp")
    if edf.empty:
        return {"error": f"Employee {employee_id} not found."}
    return {
        "employee_id":     employee_id,
        "department":      edf["department"].iloc[0],
        "total_transactions": len(edf),
        "total_spend_inr": round(float(edf["amount"].sum()), 2),
        "avg_transaction": round(float(edf["amount"].mean()), 2),
        "median_transaction": round(float(edf["amount"].median()), 2),
        "max_transaction": round(float(edf["amount"].max()), 2),
        "flagged_transactions": int((edf["anomaly_type"] != "none").sum()),
        "high_risk_transactions": int(edf["true_risk"].sum()),
        "top_vendors": edf.groupby("vendor_id")["amount"].sum()
                         .sort_values(ascending=False).head(3)
                         .reset_index().rename(columns={"amount": "total_spend"})
                         .to_dict(orient="records"),
        "recent_5": [
            {
                "transaction_id": r["transaction_id"],
                "timestamp": str(r["timestamp"]),
                "amount_inr": round(float(r["amount"]), 2),
                "risk_score": round(float(r["risk_score"]), 2),
                "anomaly_type": r["anomaly_type"],
            }
            for _, r in edf.tail(5).iterrows()
        ],
    }


def get_department_summary(department: str) -> dict:
    """Return spending summary for a department."""
    ddf = df_txn[df_txn["department"].str.lower() == department.lower()]
    if ddf.empty:
        return {"error": f"Department '{department}' not found."}
    return {
        "department":         department,
        "total_transactions": len(ddf),
        "total_spend_inr":    round(float(ddf["amount"].sum()), 2),
        "avg_transaction":    round(float(ddf["amount"].mean()), 2),
        "median_transaction": round(float(ddf["amount"].median()), 2),
        "flagged_transactions": int((ddf["anomaly_type"] != "none").sum()),
        "high_risk_transactions": int(ddf["true_risk"].sum()),
        "top_categories": ddf.groupby("category")["amount"].sum()
                            .sort_values(ascending=False).head(3)
                            .reset_index().rename(columns={"amount": "total_spend"})
                            .to_dict(orient="records"),
        "top_employees_by_spend": ddf.groupby("employee_id")["amount"].sum()
                                    .sort_values(ascending=False).head(3)
                                    .reset_index().rename(columns={"amount": "total_spend"})
                                    .to_dict(orient="records"),
    }


def get_forecast_results() -> dict:
    """Return the model comparison table from Milestone 3."""
    return {
        "forecast_models": df_forecast.reset_index().to_dict(orient="records"),
        "note": "MAE and RMSE are in INR. MAPE is percentage error.",
    }


def get_risk_summary() -> dict:
    """Return overall portfolio risk statistics."""
    total = len(df_txn)
    flagged = (df_txn["anomaly_type"] != "none").sum()
    true_risk = df_txn["true_risk"].sum()
    critical = (df_txn["risk_tier"] == "CRITICAL").sum()
    high = (df_txn["risk_tier"] == "HIGH").sum()
    exposure = df_txn[df_txn["true_risk"]]["amount"].sum()
    return {
        "total_transactions":     int(total),
        "flagged_anomalies":      int(flagged),
        "true_risk_transactions": int(true_risk),
        "critical_alerts":        int(critical),
        "high_alerts":            int(high),
        "total_spend_inr":        round(float(df_txn["amount"].sum()), 2),
        "potential_exposure_inr": round(float(exposure), 2),
        "avg_risk_score":         round(float(df_txn["risk_score"].mean()), 2),
        "top_risk_score":         round(float(df_txn["risk_score"].max()), 2),
    }


def get_trend_alerts() -> dict:
    """Return drift detection results from Milestone 3."""
    if df_trends.empty:
        return {"trend_alerts": [], "note": "No significant spending drift detected across departments or categories."}
    return {"trend_alerts": df_trends.to_dict(orient="records")}


# ─────────────────────────────────────────────────────────────
# TOOL SCHEMA — what the LLM sees when it decides which tool to call
# ─────────────────────────────────────────────────────────────
TOOLS = [
    {
        "name": "get_top_risks",
        "description": "Get the top-K highest risk score transactions. Use when asked about the biggest risks, what to investigate, or the most dangerous transactions.",
        "input_schema": {
            "type": "object",
            "properties": {
                "k": {"type": "integer", "description": "Number of top-risk transactions to return (default 10, max 50)", "default": 10}
            },
        },
    },
    {
        "name": "get_transaction",
        "description": "Get full details, risk breakdown, and SHAP explanation for a specific transaction by its ID.",
        "input_schema": {
            "type": "object",
            "properties": {
                "transaction_id": {"type": "string", "description": "The transaction ID (e.g. TX335BDA78)"}
            },
            "required": ["transaction_id"],
        },
    },
    {
        "name": "get_vendor_history",
        "description": "Get full transaction history and statistics for a specific vendor.",
        "input_schema": {
            "type": "object",
            "properties": {
                "vendor_id": {"type": "string", "description": "The vendor ID (e.g. VEN031)"}
            },
            "required": ["vendor_id"],
        },
    },
    {
        "name": "get_employee_history",
        "description": "Get full transaction history and statistics for a specific employee.",
        "input_schema": {
            "type": "object",
            "properties": {
                "employee_id": {"type": "string", "description": "The employee ID (e.g. EMP0142)"}
            },
            "required": ["employee_id"],
        },
    },
    {
        "name": "get_department_summary",
        "description": "Get spending summary for a department (Sales, Marketing, HR, Operations, Engineering, Finance).",
        "input_schema": {
            "type": "object",
            "properties": {
                "department": {"type": "string", "description": "Department name"}
            },
            "required": ["department"],
        },
    },
    {
        "name": "get_forecast_results",
        "description": "Get the forecasting model comparison results (MAE, RMSE, MAPE for Moving Average, Exponential Smoothing, ARIMA).",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "get_risk_summary",
        "description": "Get overall portfolio risk statistics: total transactions, flagged anomalies, exposure amount, risk tier breakdown.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "get_trend_alerts",
        "description": "Get spending drift detection results — departments or categories with consistently increasing spend.",
        "input_schema": {"type": "object", "properties": {}},
    },
]

# Map tool name → function
TOOL_MAP = {
    "get_top_risks":          get_top_risks,
    "get_transaction":        get_transaction,
    "get_vendor_history":     get_vendor_history,
    "get_employee_history":   get_employee_history,
    "get_department_summary": get_department_summary,
    "get_forecast_results":   get_forecast_results,
    "get_risk_summary":       get_risk_summary,
    "get_trend_alerts":       get_trend_alerts,
}

SYSTEM_PROMPT = """You are Financial Sentinel's AI analyst. You have access to tools that retrieve real financial data.

Rules:
1. ALWAYS call a tool before making any specific financial claim. Never state amounts, counts, or risk scores from memory.
2. Ground every answer in tool output. If a tool returns a number, use that exact number.
3. If you need multiple pieces of evidence (e.g. top risks + vendor history), call multiple tools.
4. Be concise. Lead with the most important finding. Use ₹ for rupee amounts.
5. If asked why something is risky, always call get_transaction to retrieve SHAP explanations.
"""


# ─────────────────────────────────────────────────────────────
# AGENT LOOP
# ─────────────────────────────────────────────────────────────
def run_agent(user_question: str, api_key: str, verbose: bool = True) -> str:
    """
    Run the Financial Sentinel agent using Gemini.
    Returns the final answer as a string.
    """

    client = genai.Client(api_key=api_key)

    # Convert existing tool definitions to Gemini function declarations
    gemini_declarations = []

    for tool in TOOLS:
        gemini_declarations.append(
            types.FunctionDeclaration(
                name=tool["name"],
                description=tool["description"],
                parameters=tool["input_schema"],
            )
        )

    gemini_tool = types.Tool(
        function_declarations=gemini_declarations
    )

    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        tools=[gemini_tool],
        max_output_tokens=1500,
    )

    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text=user_question)
            ],
        )
    ]

    if verbose:
        print(f"\n{'='*60}")
        print(f"QUESTION: {user_question}")
        print(f"{'='*60}")

    for turn in range(6):

        response = client.models.generate_content(
             model="gemini-3.6-flash",
            contents=contents,
            config=config,
        )

        # Add Gemini's response to the conversation
        contents.append(response.candidates[0].content)

        # Find any tool calls
        tool_calls = []

        for part in response.candidates[0].content.parts:
            if part.function_call:
                tool_calls.append(part.function_call)

        # No tool calls = final answer
        if not tool_calls:
            final = response.text.strip()

            if verbose:
                print(f"\nAGENT: {final}")

            return final

        # Execute each requested tool
        for tool_call in tool_calls:

            tool_name = tool_call.name
            args = dict(tool_call.args) if tool_call.args else {}

            if verbose:
                print(f"\n  [Tool call] {tool_name}({args})")

            fn = TOOL_MAP.get(tool_name)

            if fn:
                try:
                    result = fn(**args)
                except Exception as e:
                    result = {"error": str(e)}
            else:
                result = {
                    "error": f"Unknown tool: {tool_name}"
                }

            if verbose:
                preview = json.dumps(result, default=str)
                print(
                    f"  [Tool result] "
                    f"{preview[:300]}"
                    f"{'...' if len(preview) > 300 else ''}"
                )

            # Send tool result back to Gemini
            function_response = types.Part.from_function_response(
                name=tool_name,
                response={"result": result},
            )

            contents.append(
                types.Content(
                    role="user",
                    parts=[function_response],
                )
            )

    return "Agent reached maximum tool-call rounds without a final answer."


# ─────────────────────────────────────────────────────────────
# CLI — run directly to test the agent
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    api_key = os.environ.get("GEMINI_API_KEY", "")

    if not api_key:
        print("\nTo run the agent, set your Gemini API key:")
        print("  Windows: set GEMINI_API_KEY=your-key-here")
        print("  Mac/Linux: export GEMINI_API_KEY=your-key-here")
        print("\nThen run: python src/agent.py")
        print("\nRunning in OFFLINE MODE — showing tool outputs only (no LLM):\n")

        print("--- get_risk_summary() ---")
        print(json.dumps(get_risk_summary(), indent=2))

        print("\n--- get_top_risks(k=5) ---")
        print(json.dumps(get_top_risks(k=5), indent=2))

        sys.exit(0)

    # Interactive loop
    print("\nFinancial Sentinel AI Analyst")
    print("Type a question. Type 'quit' to exit.\n")

    print("Example questions:")
    print("  - What should I investigate today?")
    print("  - Why is transaction TX335BDA78 risky?")
    print("  - What is the overall risk picture?")
    print("  - How accurate were the spending forecasts?\n")

    while True:
        q = input("You: ").strip()

        if q.lower() in ("quit", "exit", "q"):
            break

        if not q:
            continue

        answer = run_agent(q, api_key)
        print(f"\nFinal Answer:\n{answer}\n")
