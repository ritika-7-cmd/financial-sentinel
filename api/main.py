"""
Financial Sentinel — FastAPI Backend
=======================================
Exposes the risk engine, alerts, forecasts, and AI agent
as a REST API. The dashboard (frontend) calls these endpoints.

Endpoints:
  GET  /                        health check
  GET  /risk-summary            overall portfolio stats
  GET  /alerts                  top-100 risk alerts
  GET  /alerts/{transaction_id} single transaction details + SHAP
  GET  /forecast                model comparison results
  GET  /trends                  drift detection alerts
  POST /analyst/query           AI agent query (requires API key)

Run locally:
  uvicorn api.main:app --reload --port 8000
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from src.agent import (
    get_risk_summary, get_top_risks, get_transaction,
    get_vendor_history, get_employee_history,
    get_department_summary, get_forecast_results,
    get_trend_alerts, run_agent
)

app = FastAPI(
    title="Financial Sentinel API",
    description="Explainable financial risk detection and AI analyst",
    version="1.0.0",
)

# Allow the dashboard (running on a different port) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Health check ──────────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "ok", "service": "Financial Sentinel API"}


# ── Risk summary ──────────────────────────────────────────────
@app.get("/risk-summary")
def risk_summary():
    return get_risk_summary()


# ── Top alerts ────────────────────────────────────────────────
@app.get("/alerts")
def alerts(k: int = 20):
    if k < 1 or k > 100:
        raise HTTPException(status_code=400, detail="k must be between 1 and 100")
    return get_top_risks(k=k)


# ── Single transaction ────────────────────────────────────────
@app.get("/alerts/{transaction_id}")
def transaction_detail(transaction_id: str):
    result = get_transaction(transaction_id)
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return result


# ── Vendor history ────────────────────────────────────────────
@app.get("/vendor/{vendor_id}")
def vendor_history(vendor_id: str):
    result = get_vendor_history(vendor_id)
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return result


# ── Employee history ──────────────────────────────────────────
@app.get("/employee/{employee_id}")
def employee_history(employee_id: str):
    result = get_employee_history(employee_id)
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return result


# ── Department summary ────────────────────────────────────────
@app.get("/department/{department}")
def department_summary(department: str):
    result = get_department_summary(department)
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return result


# ── Forecast results ──────────────────────────────────────────
@app.get("/forecast")
def forecast():
    return get_forecast_results()


# ── Trend alerts ──────────────────────────────────────────────
@app.get("/trends")
def trends():
    return get_trend_alerts()


# ── AI analyst ────────────────────────────────────────────────
class AnalystQuery(BaseModel):
    question: str
    api_key:  str   # passed from frontend; never stored server-side

@app.post("/analyst/query")
def analyst_query(body: AnalystQuery):
    if not body.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    if not body.api_key.strip():
        raise HTTPException(status_code=400, detail="API key required")
    try:
        answer = run_agent(body.question, api_key=body.api_key, verbose=False)
        return {"question": body.question, "answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
