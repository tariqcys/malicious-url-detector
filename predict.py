import json
import numpy as np
import joblib
from feature_builder import extract_features

import lzma
import joblib

with lzma.open("light_model.pkl.xz", "rb") as f:
    model = joblib.load(f)

with open("feature_columns.json", "r") as f:
    FEATURE_COLS = json.load(f)

def predict_url(url: str):
    feats = extract_features(url)
    feats = feats.reshape(1, -1)
    pred = model.predict(feats)[0]
    prob = model.predict_proba(feats)[0, 1]
    return int(pred), float(prob)
