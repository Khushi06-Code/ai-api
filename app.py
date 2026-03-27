from fastapi import FastAPI
from pydantic import BaseModel
from ml_model import predict_sepsis

from fastapi.middleware.cors import CORSMiddleware
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # allow all (easy fix)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
class PatientData(BaseModel):
    hr_mean: float
    temp_celsius_mean: float
    spo2_mean: float
    sbp_mean: float
    dbp_mean: float
    respiratory_rate_mean: float
    wbc: float
    glucose: float

@app.get("/")
def home():
    return {"message": "API working ✅"}

# ✅ POST (for docs / frontend)
@app.post("/predict")
def predict(data: PatientData):
    return predict_sepsis(data.dict())

# ✅ GET (for browser)
@app.get("/predict")
def predict_get(
    hr_mean: float,
    temp_celsius_mean: float,
    spo2_mean: float,
    sbp_mean: float,
    dbp_mean: float,
    respiratory_rate_mean: float,
    wbc: float,
    glucose: float
):
    data = {
        "hr_mean": hr_mean,
        "temp_celsius_mean": temp_celsius_mean,
        "spo2_mean": spo2_mean,
        "sbp_mean": sbp_mean,
        "dbp_mean": dbp_mean,
        "respiratory_rate_mean": respiratory_rate_mean,
        "wbc": wbc,
        "glucose": glucose
    }
    return predict_sepsis(data)
