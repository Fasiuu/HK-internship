import pandas as pd
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from langdetect import detect_langs

print("Loading data...")
df = pd.read_csv("dataset.csv")

df['Text'] = df['Text'].astype(str)

print("Preprocessing data...")

def preprocess_text(text):
    text = str(text)
    text = text.lower()
    text = text.strip()
    return text

df["cleaned_text"] = df["Text"].apply(preprocess_text)

X = df["cleaned_text"]
Y = df["Language"]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=100)

print("Transforming text into features...")

vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 4))

X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

print("Training the model...")

model = MultinomialNB()
model.fit(X_train_vectorized, Y_train)

print("Evaluating the model...")

Y_pred = model.predict(X_test_vectorized)

print("Classification Report:")
print(classification_report(Y_test, Y_pred))
print("Confusion Matrix:")
print(confusion_matrix(Y_test, Y_pred))
print("Accuracy Score:")
print(accuracy_score(Y_test, Y_pred))

# ==========================================================
# BONUS FEATURE: VISUALIZE CONFUSION MATRIX
# ==========================================================
print("Generating Confusion Matrix Plot...")
cm = confusion_matrix(Y_test, Y_pred)
labels = sorted(Y.unique())
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.title('Language Detection Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.savefig('confusion_matrix.png') # Saves the image to your folder
print("Saved plot as 'confusion_matrix.png'!")

# ==========================================================
# BONUS FEATURE: SAVE TRAINED MODEL USING PICKLE
# ==========================================================
print("Saving model to disk...")
with open("language_model.pkl", "wb") as f:
    pickle.dump(model, f)
with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)
print("Model files saved successfully!")

print("\n--- Language Detection System Ready ---")
print("Type 'quit' to exit")

while True:
    user_input = input("\nEnter text to detect language: ")

    if user_input.lower() == "quit":
        print("Exiting the language detection system. Goodbye!")
        break   

    if not user_input.strip():
        print("Please enter some text to detect the language.")
        continue

    cleaned_input = preprocess_text(user_input)
    input_vectorized = vectorizer.transform([cleaned_input])
    
    # ==========================================================
    # BONUS FEATURE: DETECT TOP 2 LANGUAGES WITH PROBABILITIES
    # ==========================================================
    probabilities = model.predict_proba(input_vectorized)[0]
    all_languages = model.classes_
    
    # Sort probabilities in descending order and get top 2 indices
    import numpy as np
    top_2_indices = np.argsort(probabilities)[::-1][:2]
    
    print("\nOur Model Predictions:")
    for idx in top_2_indices:
        lang = all_languages[idx]
        prob = probabilities[idx] * 100
        print(f"- {lang}: {prob:.2f}% confidence")

    # ==========================================================
    # BONUS FEATURE: COMPARE WITH LANGDETECT LIBRARY
    # ==========================================================
    try:
        langdetect_result = detect_langs(user_input)[0]
        print(f"Langdetect Library Prediction: {langdetect_result.lang.upper()} ({langdetect_result.prob * 100:.2f}% confidence)")
    except Exception:
        print("Langdetect Library Prediction: Could not determine language.")
