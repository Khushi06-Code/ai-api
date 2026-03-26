import joblib
import pandas as pd

model = joblib.load("model.pkl")

def predict_sepsis(features):
    # 👇 add column names


    features_df = pd.DataFrame([features])

    pred = model.predict_proba(features_df)[0][1]

    return {
    "risk_score": round(float(pred), 2),
    "status": (
        "Safe" if pred < 0.25 else
        "Low" if pred < 0.5 else
        "High" if pred < 0.75 else
        "Critical"
    )
}