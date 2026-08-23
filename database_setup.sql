-- ============================================================
-- Fake News Detection System - MySQL Database Setup
-- Run this file first: mysql -u root -p < database_setup.sql
-- ============================================================

-- 1. Create the database
CREATE DATABASE IF NOT EXISTS fake_news_db;

-- 2. Select it for use
USE fake_news_db;

-- 3. Table to store the training dataset (real + fake articles)
CREATE TABLE IF NOT EXISTS news_articles (
    article_id   INT AUTO_INCREMENT PRIMARY KEY,
    title        VARCHAR(500),
    article_text LONGTEXT NOT NULL,
    label        ENUM('REAL', 'FAKE') NOT NULL,
    source       VARCHAR(100) DEFAULT 'dataset',
    date_added   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 4. Table to log every model training run and its performance
CREATE TABLE IF NOT EXISTS model_metrics (
    metric_id   INT AUTO_INCREMENT PRIMARY KEY,
    model_name  VARCHAR(100) NOT NULL,
    accuracy    DECIMAL(6,4),
    f1_score    DECIMAL(6,4),
    trained_on  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 5. Table to store every user prediction request and result
CREATE TABLE IF NOT EXISTS predictions (
    prediction_id    INT AUTO_INCREMENT PRIMARY KEY,
    input_text        LONGTEXT NOT NULL,
    predicted_label    ENUM('REAL', 'FAKE') NOT NULL,
    confidence_score   DECIMAL(5,2),
    prediction_date    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 6. Speed up queries that filter by label (used constantly during training)
CREATE INDEX idx_label ON news_articles(label);

-- 7. Speed up lookups on prediction history by date
CREATE INDEX idx_pred_date ON predictions(prediction_date);

-- 8. Verify tables were created correctly
SHOW TABLES;
DESCRIBE news_articles;
DESCRIBE predictions;
DESCRIBE model_metrics;

-- ============================================================
-- Useful queries you'll want during the project / demo / viva
-- ============================================================

-- Count how many REAL vs FAKE articles are loaded
-- SELECT label, COUNT(*) AS total FROM news_articles GROUP BY label;

-- View the last 10 predictions made by users
-- SELECT input_text, predicted_label, confidence_score, prediction_date
-- FROM predictions ORDER BY prediction_date DESC LIMIT 10;

-- View model performance history (to show improvement over retraining)
-- SELECT model_name, accuracy, f1_score, trained_on FROM model_metrics ORDER BY trained_on DESC;

-- Find how many articles came from each source
-- SELECT source, COUNT(*) FROM news_articles GROUP BY source;
