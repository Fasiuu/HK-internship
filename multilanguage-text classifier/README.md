# Multilingual Language Detection System

> A fast, lightweight NLP pipeline leveraging character-level TF-IDF and Multinomial Naive Bayes to identify text languages with high precision.

## Features
- Custom Machine Learning Pipeline: Trained on custom datasets using scikit-learn.

Sub-word N-gram Analysis: Uses character-level n-grams (2 to 4 characters) making it robust against typos and highly accurate for short text.

Top-2 Confidence Scores: Displays the top two predicted languages alongside their exact confidence percentages.

Library Comparison: Cross-references results in real-time with the established langdetect library.

Performance Visualization: Automatically generates and saves a Seaborn confusion matrix heatmap (confusion_matrix.png) during evaluation.

Model Persistence: Saves the trained model and vectorizer as .pkl files for instant reloading without retraining.

## How To Execute
1. Set up dependencies: `pip install pandas scikit-learn matplotlib seaborn langdetect`
2. Train and validate: `python main.py`
