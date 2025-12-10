def predict_url(url):
    features = extract_features(url)

    print("RAW FEATURES:", features)
    print("RAW FEATURES SHAPE:", features.shape)

    X = pd.DataFrame([features], columns=feature_cols)

    print("DATAFRAME SHAPE:", X.shape)
    print("DATAFRAME HEAD:")
    print(X.head())

    prediction = model.predict(X)[0]
    probability = model.predict_proba(X)[0].max()

    print("PREDICTION:", prediction)
    print("PROBABILITY:", probability)

    return prediction, probability
