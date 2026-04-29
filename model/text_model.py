import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from utils.text_preprocessing import clean_text


def train_model():
    """
    Trains TF-IDF + Linear SVM model using tweets.csv
    Returns trained model and vectorizer
    """

    data = pd.read_csv("tweets.csv")

    # Ensure correct columns
    data["processed"] = data["Tweet"].apply(clean_text)

    # Map labels exactly as used in app.py
    data["label"] = data["Text Label"].map({
        "Predator": 1,
        "Non-Predator": 0
    })

    vectorizer = TfidfVectorizer(
        max_features=6000,
        ngram_range=(1, 2),
        stop_words="english"
    )

    X = vectorizer.fit_transform(data["processed"])
    y = data["label"]

    model = LinearSVC(class_weight="balanced")
    model.fit(X, y)

    return model, vectorizer


def predict_text(model, vectorizer, text):
    """
    Predicts label for a single line of text
    Returns:
    - label (Predator / Non-Predator)
    - processed text
    - short summary (first few words)
    """

    processed = clean_text(text)

    if not processed.strip():
        return "Non-Predator", "", ""

    vector = vectorizer.transform([processed])
    prediction = model.predict(vector)[0]

    label = "Predator" if prediction == 1 else "Non-Predator"

    # Short summary for reporting (optional use)
    summary = " ".join(processed.split()[:8])

    return label, processed, summary
