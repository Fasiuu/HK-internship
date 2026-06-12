# Multilingual Language Detection System

An end-to-end Machine Learning pipeline utilizing character-level features and a Multinomial Naive Bayes model to dynamically identify the native language of provided textual data strings.

## Features
- **Engine Core:** Character-level TF-IDF Vectorization ($N$-grams: 2 to 4) optimizing calculations for short sentences or unique unicode character sets.
- **Visual Analytics:** Generates confusion matrices mapping classification performance.
- **Top-2 Evaluation:** Reports secondary probable classification options alongside confidence score indicators.
- **Benchmarking Integration:** Cross-compares real-time inferences with the `langdetect` Python library.
- **Batch Processor:** Accepts user-defined structural target documents (CSVs) for swift multi-row automated profiling.

## How To Execute
1. Set up dependencies: `pip install pandas scikit-learn matplotlib seaborn langdetect`
2. Train and validate: `python main.py`
