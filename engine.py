"""Core logic: risk profiling, portfolio optimisation, goal planning, timeline, rebalancing."""
import json, os, joblib, numpy as np, pandas as pd
from datetime import date
from scipy.optimize import minimize

HERE = os.path.dirname(__file__)
try: ML = joblib.load(os.path.join(HERE, "models/risk_model.joblib"))
except Exception: ML = None

# ---------------- Risk profiling (formulas follow IDea_lab.ipynb) ----------------
INC = {"Less than 3 Lakhs": 30, "3 to 6 Lakhs": 50, "6 to 10 lakhs": 70, "More than 10 lakhs": 90}
LABELS = ["Very Conservative", "Conservative", "Moderate", "Moderately Aggressive", "Aggressive"]
BINS = [0, 40, 55, 70, 82, 100]
QUIZ = [  # financial-literacy items taken from the survey instrument (index of right answer)
 ("Rs.100 in a savings account at 20% p.a., untouched for 5 years. You would have…", ["Less than Rs.200", "Exactly Rs.200", "More than Rs.200", "Don't know"], 2),
 ("Savings earn 1% p.a. while inflation is 2%. After 1 year you can buy…", ["More than today", "Exactly the same", "Less than today", "Don't know"], 2),
 ("Income doubles and all prices double. You can buy…", ["More", "The same", "Less", "Don't know"], 1),
 ("Who is richer: Rs.10,000 inherited today, or Rs.10,000 three years from now?", ["Today's heir", "Future heir", "Both equal", "Don't know"], 0),
 ("Spreading money across different assets makes the risk of loss…", ["Increase", "Decrease", "Stay the same", "Don't know"], 1),
 ("Over 10-20 years, which asset has usually given the highest return?", ["Savings account", "Bonds", "Stocks", "Gold"], 2),
 ("When interest rates fall, bond prices usually…", ["Rise", "Fall", "Stay the same", "Don't know"], 0)]
BIAS = ["loss_aversion", "panic", "overconfidence", "fomo", "herding", "anchoring"]  # Likert 1-5 -> 0-100

def risk_profile(a):
    qz = a.get("quiz") or {}   # {question_index: chosen_option}; any subset of the 7 items is scored
    qz = qz if isinstance(qz, dict) else dict(enumerate(qz))
    fl = 100 * sum(1 for i, c in qz.items() if QUIZ[int(i)][2] == c) / len(qz) if qz else 50.0
    b = {k: float((a.get("bias") or {}).get(k, 3)) * 20 for k in BIAS}
    cap = 0.5 * INC[a["income"]] + 0.5 * fl
    adj = []
    if a.get("emergency_months", 6) < 3: cap -= 10; adj.append("Emergency fund < 3 months (-10 capacity)")
    if a.get("dependents", 0) >= 3: cap -= 5; adj.append("3+ dependants (-5 capacity)")
    if a["age"] >= 55: cap -= 10; adj.append("Age 55+ (-10 capacity)")
    elif a["age"] <= 30: cap += 3; adj.append("Long earning runway (+3 capacity)")
    cap = float(np.clip(cap, 0, 100))
    tol = float(np.clip(100 - .35 * b["loss_aversion"] - .25 * b["panic"] + .20 * b["overconfidence"] + .15 * b["fomo"], 15, 95))
    comb = .5 * cap + .5 * tol
    label = LABELS[int(np.digitize(comb, BINS[1:-1]))]
    ml = None
    if ML is not None:
        occ = a.get("occupation", "Salaried"); occ = occ if occ in ("Salaried", "Self Employed") else "Other"
        row = pd.DataFrame([{"gender": a.get("gender", "Male"), "education": a.get("education", "Graduate"), "occupation": occ,
            "annual_income_bracket": a["income"], "marital_status": a.get("marital_status", "Single"),
            "work_experience": a.get("work_experience", "2 to 5 years"), "objective_fl_pct": fl,
            "overconfidence_score": b["overconfidence"], "anchoring_bias_score": b["anchoring"], "loss_aversion_score": b["loss_aversion"],
            "herding_bias_score": b["herding"], "cultural_influence_score": 68.0, "market_crash_panic_score": b["panic"],
            "bull_market_euphoria_fomo": b["fomo"], "risk_coverage_motive": 72.0}])
        pr = ML.predict_proba(row)[0]
        ml = {"prediction": str(ML.classes_[pr.argmax()]), "probabilities": {str(c): round(float(p), 3) for c, p in zip(ML.classes_, pr)}}
    return {"label": label, "score": round(comb, 1), "capacity": round(cap, 1), "tolerance": round(tol, 1), "fl_pct": round(fl, 1),
            "biases": {k: round(v) for k, v in b.items()}, "adjustments": adj, "ml_crosscheck": ml,
            "coaching": coaching(b, fl, label)}

def coaching(b, fl, label):
    t = []
    if b["panic"] >= 80: t.append("High crash-panic: automate SIPs and pre-commit to NOT selling during >15% drawdowns.")
    if b["fomo"] >= 80: t.append("Strong FOMO: cap tactical/thematic bets at 5% and rebalance only on rule-based triggers.")
    if b["loss_aversion"] >= 80: t.append("High loss aversion: the plan keeps a larger debt/liquid buffer and shows downside (P10) outcomes.")
    if b["herding"] >= 80: t.append("Herding tendency: avoid tips; follow the written allocation, not the crowd.")
    if b["overconfidence"] >= 80: t.append("Overconfidence: prefer low-cost index funds over stock-picking; review performance vs benchmark annually.")
    if fl < 60: t.append("Financial-literacy score is modest: start with index funds and simple instruments.")
    return t or ["Behaviour profile is balanced; stay disciplined with periodic reviews."]

# ---------------- Portfolio optimisation (long-only mean-variance with profile & horizon bounds) ----------------
ASSETS = ["Large-cap equity", "Mid/small-cap equity", "Debt (short/medium)", "Gold", "Liquid/emergency"]
MU = np.array([.12, .145, .07, .085, .06]); SD = np.array([.16, .22, .04, .14, .01])   # long-run INR assumptions (editable)
CORR = np.array([[1, .85, .05, .0, .0], [.85, 1, .0, .0, .0], [.05, 0, 1, .1, .3], [0, 0, .1, 1, .1], [0, 0, .3, .1, 1]])
COV = CORR * np.outer(SD, SD)
PROF = {"Very Conservative": (12, .20, 0), "Conservative": (8, .35, .05), "Moderate": (5, .55, .15),
        "Moderately Aggressive": (3.2, .75, .25), "Aggressive": (2.2, .90, .35)}  # (risk aversion, max equity, max mid/small)

def horizon_cap(y):
    for lim, c in [(1, 0), (2, .10), (3, .20), (5, .40), (7, .60), (10, .75)]:
        if y <= lim: return c
    return 1.0

_cache = {}
def optimise(label, years):
    lam, eq, mid = PROF[label]; cap = min(eq, horizon_cap(years)); key = (label, cap)
    if key in _cache: return _cache[key]
    liq_min = .10 if years < 3 else .05
    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1}, {"type": "ineq", "fun": lambda w: cap - w[0] - w[1]}]
    bnds = [(0, cap), (0, min(mid, cap) if cap > .4 else 0), (.05, 1), (0, .15), (liq_min, 1)]
    r = minimize(lambda w: -(MU @ w - lam / 2 * w @ COV @ w), [.2] * 5, bounds=bnds, constraints=cons, method="SLSQP")
    w = np.round(np.clip(r.x, 0, None) * 20) / 20; w[2] += 1 - w.sum()   # snap to 5% steps, debt absorbs remainder
    out = {"weights": {a: round(float(x), 2) for a, x in zip(ASSETS, w)}, "exp_return": round(float(MU @ w), 4),
           "volatility": round(float(np.sqrt(w @ COV @ w)), 4), "equity_cap": cap}
    _cache[key] = out; return out

# ---------------- Goals ----------------
INFL = {"Education": .08, "Home": .06, "Retirement": .06, "Wedding": .07, "Vehicle": .05, "Travel": .06, "Emergency": .06, "Other": .06}

def det_fv(corpus, sip0, mu, n, step):
    v = corpus * (1 + mu) ** n
    for t in range(int(np.ceil(n))): v += 12 * sip0 * (1 + step) ** t * (1 + mu) ** (n - t - .5)
    return v

def plan_goal(g, prof, step=.05, rng=np.random.default_rng(7)):
    n = max(float(g["years"]), .25); inf = INFL.get(g.get("category", "Other"), .06)
    target = g["target_today"] * (1 + inf) ** n
    pf = optimise(prof, n); mu, sd = pf["exp_return"], pf["volatility"]
    corpus, sip = g.get("current_corpus", 0), g.get("monthly_sip", 0)
    base = max(0, (target - det_fv(corpus, 0, mu, n, 0)) / det_fv(0, 1, mu, n, step))   # at expected return only
    N, Y = 4000, int(np.ceil(n)); s_ = np.sqrt(np.log(1 + (sd / (1 + mu)) ** 2)); m_ = np.log(1 + mu) - s_ ** 2 / 2
    R = np.exp(rng.normal(m_, s_, (N, Y)))
    A = np.full(N, float(corpus)); B = np.zeros(N); paths = [A.copy()]
    for t in range(Y):  # V = A + sip*B is linear in SIP, so we track both parts
        A = A * R[:, t]; B = B * R[:, t] + 12 * (1 + step) ** t * np.sqrt(R[:, t]); paths.append(A + sip * B)
    need = max(0, float(np.percentile((target - A) / B, 75)))                                 # 75% confidence SIP
    P = np.array(paths); prob = float((A + sip * B >= target).mean())
    pct = lambda q: [round(float(x)) for x in np.percentile(P, q, axis=1)]
    # glide path (equity share by years remaining)
    glide, last = [], None
    for t in range(Y + 1):
        left = n - t; w = optimise(prof, max(left, .1))["weights"]; eq = round(w[ASSETS[0]] + w[ASSETS[1]], 2)
        if eq != last: glide.append({"year_offset": t, "years_left": round(left, 1), "equity": eq, "weights": w}); last = eq
    return {**g, "inflation": inf, "future_cost": round(target), "portfolio": pf, "required_sip": round(need), "baseline_sip": round(base), "current_sip": sip,
            "step_up": step, "success_prob": round(prob, 3), "status": "On track" if prob >= .75 else ("At risk" if prob >= .5 else "Off track"),
            "projection": {"years": list(range(Y + 1)), "p10": pct(10), "p50": pct(50), "p90": pct(90)}, "glide_path": glide}

def add_months(d, m):
    y, mo = divmod(d.month - 1 + m, 12); day = min(d.day, 28); return date(d.year + y, mo + 1, day)

def timeline(plans, prof, today=None):
    today = today or date.today(); ev = []
    for p in plans:
        n = float(p["years"]); end = add_months(today, round(n * 12)); nm = p["name"]
        ev.append(dict(date=today.isoformat(), goal=nm, type="start", title=f"Start SIP ₹{p['required_sip']:,}/mo", detail=f"Portfolio: {', '.join(f'{k} {int(v*100)}%' for k, v in p['portfolio']['weights'].items() if v)}"))
        for y in range(1, int(n) + 1):
            d = add_months(today, 12 * y)
            ev.append(dict(date=d.isoformat(), goal=nm, type="review", title="Annual review + SIP step-up", detail=f"Raise SIP {int(p['step_up']*100)}%; re-check probability of success."))
        for gp in p["glide_path"][1:]:
            d = add_months(today, round(gp["year_offset"] * 12))
            ev.append(dict(date=d.isoformat(), goal=nm, type="derisk", title=f"De-risk: equity → {int(gp['equity']*100)}%", detail=f"{gp['years_left']} yrs left. Shift gains from equity to debt/liquid."))
        if n > 1: ev.append(dict(date=add_months(end, -12).isoformat(), goal=nm, type="derisk", title="Move 100% of corpus to safe assets", detail="12 months to target: lock in gains, stop equity SIP."))
        ev.append(dict(date=end.isoformat(), goal=nm, type="goal", title=f"🎯 {nm}: target ₹{p['future_cost']:,}", detail="Redeem and deploy funds."))
    for q in range(4):
        ev.append(dict(date=add_months(today, 3 * (q + 1)).isoformat(), goal="All", type="rebalance", title="Quarterly rebalance check", detail="Rebalance if any asset drifts > 5 pp from target."))
    ev.sort(key=lambda e: e["date"])
    for e in ev: e["days_away"] = (date.fromisoformat(e["date"]) - today).days
    return ev

def reminders(ev, prof):
    up = [e for e in ev if 0 <= e["days_away"] <= 90]
    return [{"urgency": "soon" if e["days_away"] <= 30 else "upcoming", **e} for e in up] or [{"urgency": "info", "title": "No actions due in the next 90 days", "detail": "", "date": "", "goal": ""}]

def full_plan(req):
    prof = req["risk"]["label"] if isinstance(req.get("risk"), dict) else req["risk_label"]
    plans = [plan_goal(g, prof, req.get("step_up", .05)) for g in req["goals"]]
    surplus = req.get("monthly_income", 0) - req.get("monthly_expenses", 0)
    need = sum(p["required_sip"] for p in plans); ev = timeline(plans, prof)
    order = sorted(plans, key=lambda p: (p.get("priority", 2), p["years"])); left = surplus; funding = []
    for p in order:
        give = min(left, p["required_sip"]); left -= give; funding.append({"goal": p["name"], "allocated": round(give), "required": p["required_sip"]})
    return {"goals": plans, "timeline": ev, "reminders": reminders(ev, prof), "total_required_sip": need, "surplus": surplus,
            "feasible": need <= surplus, "funding_order": funding}

# ---------------- Dynamic rebalancing ----------------
def rebalance(holdings, target, new_cash=0, threshold=.05, drawdown=0, profile="Moderate"):
    tw = dict(target)
    if drawdown >= 15 and PROF[profile][1] >= .55:       # tactical tilt: buy the dip within risk limits
        tilt = min(.05, drawdown / 400); tw[ASSETS[0]] += tilt; tw[ASSETS[2]] = max(0, tw[ASSETS[2]] - tilt)
    tot = sum(holdings.values()) + new_cash; rows, drift_flag = [], False
    for a in ASSETS:
        cur = holdings.get(a, 0); w = cur / tot if tot else 0; d = w - tw.get(a, 0); drift_flag |= abs(d) > threshold
        rows.append({"asset": a, "current": round(cur), "current_w": round(w, 3), "target_w": round(tw.get(a, 0), 3), "drift": round(d, 3), "trade": round(tw.get(a, 0) * tot - cur)})
    if new_cash and drift_flag:  # prefer routing fresh cash to under-weights before selling anything
        pass
    return {"needs_rebalance": drift_flag, "total": round(tot), "rows": rows, "tactical_tilt": drawdown >= 15,
            "advice": ("Drift exceeds %d pp: execute trades below (use new cash first to avoid tax)." % int(threshold * 100)) if drift_flag else "Portfolio is within tolerance; no action needed."}
