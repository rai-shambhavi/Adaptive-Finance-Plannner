# Adaptive Wealth Planner
```
pip install -r requirements.txt
python train_model.py          # optional: retrains models/risk_model.joblib from data/
uvicorn app:app --reload       # open http://localhost:8000
```
- `engine.py` - risk profiling (capacity x tolerance, from IDea_lab.ipynb), mean-variance optimiser with profile/horizon limits, Monte-Carlo goal planner, glide path, timeline, reminders, rebalancing.
- `app.py` - FastAPI endpoints: /api/questionnaire, /api/risk-profile, /api/plan, /api/rebalance, /api/model-card.
- `static/index.html` - frontend (no build step).
Return/volatility assumptions live in `engine.py` (MU, SD, CORR) - edit to your own views. Educational tool, not investment advice.
