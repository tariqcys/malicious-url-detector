import joblib
import json
import pandas as pd
import lzma
from feature_builder import extract_features

# ✅ تحميل أسماء الخصائص بالترتيب الصحيح
with open("feature_columns.json", "r") as f:
    feature_cols = json.load(f)

# ✅ تحميل المودل المضغوط بشكل صحيح
with lzma.open("light_model.pkl.xz", "rb") as f:
    model = joblib.load(f)

def predict_url(url):
    features = extract_features(url)

    # ✅ أهم سطر: ربط الخصائص مع أسماء الأعمدة الصحيحة
    X = pd.DataFrame([features], columns=feature_cols)

    prediction = model.predict(X)[0]
    probability = model.predict_proba(X)[0].max()

    return prediction, probability
