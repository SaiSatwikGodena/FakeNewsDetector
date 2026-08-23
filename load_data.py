"""
load_data.py
------------
Loads a labeled news dataset (CSV files) into the MySQL 'news_articles' table.

Expected input: two CSV files, e.g. from the Kaggle "Fake and Real News" dataset:
    Fake.csv  -> columns: title, text
    True.csv  -> columns: title, text

Usage:
    python load_data.py --fake Fake.csv --real True.csv
"""

import argparse
import pandas as pd
import mysql.connector
from db_config import DB_CONFIG


def get_connection():
    return mysql.connector.connect(**DB_CONFIG)


def load_csv_to_mysql(csv_path, label, source="dataset", batch_size=500):
    """Reads a CSV with 'title' and 'text' columns and bulk-inserts rows
    into news_articles with the given label ('REAL' or 'FAKE')."""

    df = pd.read_csv(csv_path)

    # Basic cleanup: drop empty rows, keep only needed columns
    df = df.dropna(subset=["text"])
    if "title" not in df.columns:
        df["title"] = ""

    conn = get_connection()
    cursor = conn.cursor()

    insert_query = """
        INSERT INTO news_articles (title, article_text, label, source)
        VALUES (%s, %s, %s, %s)
    """

    rows = [
        (str(row["title"])[:500], str(row["text"]), label, source)
        for _, row in df.iterrows()
    ]

    # Insert in batches for speed
    for i in range(0, len(rows), batch_size):
        batch = rows[i:i + batch_size]
        cursor.executemany(insert_query, batch)
        conn.commit()
        print(f"Inserted rows {i} to {i + len(batch)} for label={label}")

    cursor.close()
    conn.close()
    print(f"Done loading {csv_path} -> {len(rows)} rows as '{label}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load fake/real news CSVs into MySQL")
    parser.add_argument("--fake", required=True, help="Path to Fake.csv")
    parser.add_argument("--real", required=True, help="Path to True.csv")
    args = parser.parse_args()

    load_csv_to_mysql(args.fake, label="FAKE")
    load_csv_to_mysql(args.real, label="REAL")
