"""Train the risk-profile classifier from the unified dataset built in IDea_lab.ipynb.
Run: python train_model.py   ->  models/risk_model.joblib + models/model_card.json"""
import json, joblib, pandas as pd, numpy as np
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

df = pd.read_excel("data/Unified_Investor_Behavioral_Profiling_Dataset.xlsx", sheet_name="Unified_Master_Dataset")
df["occupation"] = df["occupation"].where(df["occupation"].isin(["Salaried", "Self Employed"]), "Other")
TARGET = "final_risk_profile"
# capacity/tolerance scores are direct antecedents of the target -> excluded (leakage, as in the notebook)
CAT = ["gender", "education", "occupation", "annual_income_bracket", "marital_status", "work_experience"]
NUM = ["objective_fl_pct", "overconfidence_score", "anchoring_bias_score", "loss_aversion_score",
       "herding_bias_score", "cultural_influence_score", "market_crash_panic_score",
       "bull_market_euphoria_fomo", "risk_coverage_motive"]
X, y = df[CAT + NUM], df[TARGET].astype(str)

pipe = Pipeline([
    ("prep", ColumnTransformer([("n", StandardScaler(), NUM),
                                ("c", OneHotEncoder(handle_unknown="ignore"), CAT)])),
    ("clf", RandomForestClassifier(n_estimators=300, max_depth=5, min_samples_leaf=8,
                                   class_weight="balanced", random_state=42))])

cv = cross_validate(pipe, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=42),
                    scoring=["f1_macro", "balanced_accuracy", "accuracy"])
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
pipe.fit(Xtr, ytr)
rep = classification_report(yte, pipe.predict(Xte), output_dict=True, zero_division=0)
pipe.fit(X, y)  # final model on all data

card = {"n_rows": len(df), "class_counts": y.value_counts().to_dict(),
        "cv_f1_macro": round(cv["test_f1_macro"].mean(), 3), "cv_balanced_acc": round(cv["test_balanced_accuracy"].mean(), 3),
        "cv_accuracy": round(cv["test_accuracy"].mean(), 3), "holdout_report": rep,
        "feature_importance": dict(sorted(zip(pipe["prep"].get_feature_names_out(), pipe["clf"].feature_importances_.round(4).tolist()),
                                          key=lambda t: -t[1])[:10]),
        "note": "Labels are rule-derived (capacity x tolerance) and behavioural scores are stratum means (gender x experience), "
                "so this model mirrors the rule engine; it is a cross-check, not independent validation."}
joblib.dump(pipe, "models/risk_model.joblib"); json.dump(card, open("models/model_card.json", "w"), indent=2, default=float)
print({k: card[k] for k in ["cv_f1_macro", "cv_balanced_acc", "cv_accuracy"]}); print(card["feature_importance"])
