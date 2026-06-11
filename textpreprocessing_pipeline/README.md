# Text Preprocessing Pipeline with Custom Tokenization

A highly modular, production-inspired text preprocessing pipeline engineered in Python using `NLTK` and Native Regular Expressions (`re`). Built for preparing noisy, raw textual data for downstream machine learning and natural language processing tasks.

##  Features Supported
* **Text Scrubbing:** Lowercasing, structural spacing optimization, stripping of punctuation/special symbols.
* **Custom Regex Engine:** Custom-built tokenization layer utilizing isolated boundaries (`\b\w+\b`).
* **Intelligent Morphological Reduction:** Lemmatizing with contextual POS hint-checks alongside standard Porter Stemming comparisons.
* **Dual Interface Routing:** Integrated interactive command-line interface (CLI) to process singular live queries or batch operations straight to `.csv`.

##  How To Run
1. Install dependencies:
   ```bash
   pip install nltk pandas
   
2. Run program using command
   ```bash
    Python main.py
