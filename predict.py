import joblib
import json
import pandas as pd
import lzma
import feature_builder

with open("feature_columns.json", "r") as f:
    feature_cols = json.load(f)

with lzma.open("light_model.pkl.xz", "rb") as f:
    model = joblib.load(f)

def predict_url(url):
    features = feature_builder.extract_features(url)

    print("RAW FEATURES:", features)
    print("RAW FEATURES SHAPE:", features.shape)

    X = pd.DataFrame([features], columns=feature_cols)

    print("DATAFRAME SHAPE:", X.shape)
    print(X.head())

    prediction = model.predict(X)[0]
    probability = model.predict_proba(X)[0].max()

    print("PREDICTION:", prediction)
    print("PROBABILITY:", probability)

    return prediction, probability
