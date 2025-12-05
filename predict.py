import json
import numpy as np
import joblib
from feature_builder import extract_features

model = joblib.load("model.pkl")

with open("feature_columns.json", "r") as f:
    FEATURE_COLS = json.load(f)

def predict_url(url: str):
    feats = extract_features(url)
    feats = feats.reshape(1, -1)
    pred = model.predict(feats)[0]
    prob = model.predict_proba(feats)[0, 1]
    return int(pred), float(prob)
