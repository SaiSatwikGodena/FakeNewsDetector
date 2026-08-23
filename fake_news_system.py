"""
fake_news_system.py
--------------------
Core Fake News Detection system.

Pipeline:
  1. Pull labeled articles OUT of MySQL (news_articles table)
  2. Preprocess text (lowercase, strip punctuation, remove stopwords)
  3. Vectorize with TF-IDF
  4. Train a Logistic Regression classifier
  5. Evaluate (accuracy, F1) and log the run INTO MySQL (model_metrics table)
  6. Take live user input, predict REAL/FAKE, and log it INTO MySQL (predictions table)

Run:
    python fake_news_system.py
"""

import re
import string
import pickle

import pandas as pd
import mysql.connector
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
import matplotlib.pyplot as plt

from db_config import DB_CONFIG

nltk.download("stopwords", quiet=True)
STOPWORDS = set(stopwords.words("english"))


# ------------------------------------------------------------------
# MySQL helpers
# ------------------------------------------------------------------
def get_connection():
    return mysql.connector.connect(**DB_CONFIG)


def fetch_dataset_from_mysql():
    """Pulls all labeled articles from MySQL into a pandas DataFrame."""
    conn = get_connection()
    query = "SELECT article_text AS text, label FROM news_articles"
    df = pd.read_sql(query, conn)
    conn.close()
    if df.empty:
        raise ValueError(
            "news_articles table is empty. Run load_data.py first to populate MySQL."
        )
    return df


def save_metrics_to_mysql(model_name, accuracy, f1):
    conn = get_connection()
    cursor = conn.cursor()
    query = """
        INSERT INTO model_metrics (model_name, accuracy, f1_score)
        VALUES (%s, %s, %s)
    """
    cursor.execute(query, (model_name, float(accuracy), float(f1)))
    conn.commit()
    cursor.close()
    conn.close()


def save_prediction_to_mysql(text, label, confidence_pct):
    conn = get_connection()
    cursor = conn.cursor()
    query = """
        INSERT INTO predictions (input_text, predicted_label, confidence_score)
        VALUES (%s, %s, %s)
    """
    cursor.execute(query, (text, label, float(confidence_pct)))
    conn.commit()
    cursor.close()
    conn.close()


# ------------------------------------------------------------------
# Preprocessing
# ------------------------------------------------------------------
def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 1]
    return " ".join(tokens)


# ------------------------------------------------------------------
# Training
# ------------------------------------------------------------------
def train_and_save_model():
    print("Fetching training data from MySQL...")
    df = fetch_dataset_from_mysql()
    print(f"Loaded {len(df)} articles from news_articles table.")

    print("Preprocessing text...")
    df["clean_text"] = df["text"].apply(preprocess_text)

    X = df["clean_text"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    vectorizer = TfidfVectorizer(max_df=0.7, min_df=2, ngram_range=(1, 2))
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    print("Training Logistic Regression model...")
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_tfidf, y_train)

    y_pred = model.predict(X_test_tfidf)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, pos_label="FAKE")

    print("\n--- Evaluation ---")
    print(classification_report(y_test, y_pred))
    print(f"Accuracy: {acc:.4f}  |  F1-score: {f1:.4f}")

    print("Logging metrics to MySQL...")
    save_metrics_to_mysql("LogisticRegression_TFIDF", acc, f1)

    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open("vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)

    plot_confusion_style_bar(y_test, y_pred)

    return model, vectorizer


def plot_confusion_style_bar(y_test, y_pred):
    """Simple bar chart comparing actual vs predicted label counts."""
    actual_counts = pd.Series(y_test).value_counts()
    pred_counts = pd.Series(y_pred).value_counts()

    labels = sorted(set(list(actual_counts.index) + list(pred_counts.index)))
    actual_vals = [actual_counts.get(l, 0) for l in labels]
    pred_vals = [pred_counts.get(l, 0) for l in labels]

    x = range(len(labels))
    plt.figure(figsize=(6, 4))
    plt.bar([i - 0.2 for i in x], actual_vals, width=0.4, label="Actual")
    plt.bar([i + 0.2 for i in x], pred_vals, width=0.4, label="Predicted")
    plt.xticks(list(x), labels)
    plt.ylabel("Count")
    plt.title("Actual vs Predicted Label Distribution (Test Set)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("evaluation_chart.png")
    print("Saved evaluation chart to evaluation_chart.png")


# ------------------------------------------------------------------
# Prediction
# ------------------------------------------------------------------
def load_saved_model():
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    return model, vectorizer


def predict_news(text, model, vectorizer):
    clean = preprocess_text(text)
    vec = vectorizer.transform([clean])
    label = model.predict(vec)[0]
    confidence = model.predict_proba(vec).max() * 100
    save_prediction_to_mysql(text, label, confidence)
    return label, confidence


# ------------------------------------------------------------------
# CLI entry point
# ------------------------------------------------------------------
def main():
    model, vectorizer = train_and_save_model()

    print("\n=== Fake News Detection System ===")
    print("Type a news headline or article. Type 'exit' to quit.\n")

    while True:
        user_input = input("Enter news text: ").strip()
        if user_input.lower() == "exit":
            break
        if not user_input:
            continue

        label, confidence = predict_news(user_input, model, vectorizer)
        result = "Real News" if label == "REAL" else "Fake News"
        print(f"--> {result}  (confidence: {confidence:.2f}%)\n")


if __name__ == "__main__":
    main()
