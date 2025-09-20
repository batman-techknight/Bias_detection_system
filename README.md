# Bias_detection_system
NLP mini project
🧠 Bias Detection System
📌 Overview

>The Bias Detection System is a Natural Language Processing (NLP) mini project designed to identify whether a given text contains biased or unbiased statements. It uses TF-IDF vectorization and a Logistic Regression classifier to detect linguistic bias.

    🎯 Objectives

>Detect biased vs. unbiased sentences in text data

>Apply preprocessing (stopword removal, stemming, tokenization)

>Train and evaluate a machine learning model

>Provide an interactive function to test custom sentences

    📂 Dataset

A custom 100-line dataset with balanced biased and unbiased samples

label = 1 → Biased

label = 0 → Not Biased

    Example:

text	label
Women are not good at driving cars.	1
AI can help improve healthcare systems.	0
⚙️ Workflow

    Data Preprocessing

Lowercasing

Removing special characters

Stopword removal

Word stemming

Feature Extraction

Convert text into numerical features using TF-IDF Vectorizer

Model Training

Logistic Regression classifier

Evaluation

Accuracy score

Classification report (precision, recall, F1-score)

Confusion matrix heatmap

Prediction Function

Custom input sentences can be tested for bias detection

    📊 Results

Accuracy achieved: ~80–90% (depends on dataset split)

Confusion matrix visualization for model performance

Real-time prediction function for bias detection

    🚀 Applications

Detecting biased language in news articles

Monitoring social media posts for fairness

Assisting researchers in analyzing biased content

Building ethical AI systems
