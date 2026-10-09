"""FastAPI backend. Run: uvicorn app:app --reload  ->  http://localhost:8000"""
import json, os
from fastapi import FastAPI, Body
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import engine as E

app = FastAPI(title="Adaptive Financial Planning & Portfolio Optimisation")
H = os.path.dirname(__file__)

@app.get("/api/questionnaire")
def questionnaire(): return {"quiz": [{"q": q, "options": o} for q, o, _ in E.QUIZ], "bias": E.BIAS, "assets": E.ASSETS, "profiles": E.LABELS}
@app.post("/api/risk-profile")
def risk(a: dict = Body(...)):
    r = E.risk_profile(a); r["allocation"] = E.optimise(r["label"], 15); return r
@app.post("/api/plan")
def plan(req: dict = Body(...)): return E.full_plan(req)
@app.post("/api/rebalance")
def rebal(req: dict = Body(...)): return E.rebalance(req["holdings"], req["target"], req.get("new_cash", 0), req.get("threshold", .05), req.get("drawdown", 0), req.get("profile", "Moderate"))
@app.get("/api/model-card")
def card(): return json.load(open(os.path.join(H, "models/model_card.json")))
@app.get("/")
def index(): return FileResponse(os.path.join(H, "static/index.html"))
app.mount("/static", StaticFiles(directory=os.path.join(H, "static")), name="static")
