from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "model" / "best_random_forest.pkl"

model = joblib.load(MODEL_PATH)


# --------------------------------------------------
# FastAPI application
# --------------------------------------------------

app = FastAPI(
    title="Football Match Prediction API",
    description="API for predicting football match outcomes using a trained Random Forest model.",
    version="1.0.0"
)


# --------------------------------------------------
# Input schema
# --------------------------------------------------

class MatchInput(BaseModel):

    league: str
    season: str
    div: str

    hometeam: str
    awayteam: str
    referee: str

    kickoff_hour: float
    time_of_day: str

    shot_accuracy_away: float
    total_shots: float
    total_shots_on_target: float
    total_fouls: float
    total_corners: float

    match_month: int
    match_dayofweek: int


# --------------------------------------------------
# Prediction endpoint
# --------------------------------------------------

@app.post("/predict")
def predict_match(match: MatchInput):

    # Convert request data into DataFrame
    input_data = pd.DataFrame([match.model_dump()])

    # Make prediction
    prediction = model.predict(input_data)[0]

    # Get probabilities
    probabilities = model.predict_proba(input_data)[0]

    # Map class names to probabilities
    probability_dict = {
        str(cls): round(float(prob), 4)
        for cls, prob in zip(model.classes_, probabilities)
    }

    # Human-readable prediction
    result_labels = {
        "H": "Home Win",
        "D": "Draw",
        "A": "Away Win"
    }

    return {
        "prediction": str(prediction),
        "prediction_label": result_labels.get(
            str(prediction),
            "Unknown"
        ),
        "probabilities": probability_dict
    }


# --------------------------------------------------
# Health check
# --------------------------------------------------

@app.get("/")
def root():

    return {
        "message": "Football Match Prediction API",
        "status": "running"
    }