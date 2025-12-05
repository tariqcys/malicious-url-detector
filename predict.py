import json
import numpy as np
import joblib
import zipfile
from feature_builder import extract_features

# فتح ملف ZIP وقراءة model.pkl منه مباشرة
with zipfile.ZipFile("model.zip", "r") as z:
    with z.open("model.pkl") as f:
        model = joblib.load(f)

# تحميل ترتيب الخصائص
with open("feature_columns.json", "r") as f:
    FEATURE_COLS = json.load(f)

def predict_url(url: str):
    feats = extract_features(url)
    feats = feats.reshape(1, -1)
    pred = model.predict(feats)[0]
    prob = model.predict_proba(feats)[0, 1]
    return int(pred), float(prob)
