import pandas as pd 
import string 
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = text.split()
    cleaned_words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return ' '.join(cleaned_words)

df = pd.read_csv('dataset.csv')
df['cleaned_text'] = df['text'].apply(preprocess_text)

X = df['cleaned_text']
Y = df['sentiment']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

model = LogisticRegression()
model.fit(X_train_tfidf, Y_train)

Y_pred = model.predict(X_test_tfidf)
labels = sorted(df['sentiment'].unique())
cm = confusion_matrix(Y_test, Y_pred, labels=labels)

plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.title('Sentiment Classifier Confusion Matrix')
plt.ylabel('Actual Sentiment')
plt.xlabel('Predicted Sentiment')
plt.savefig('confusion_matrix.png')  
print("✅ Confusion matrix plot saved as 'confusion_matrix.png'!\n")

print("--- Model Accuracy ---")
print(f"Accuracy: {accuracy_score(Y_test, Y_pred) * 100:.2f}%\n")

print("--- Classification Report ---")
print(classification_report(Y_test, Y_pred, zero_division=0))

print("================================================")
print("     MULTI-CLASS SENTIMENT ANALYSIS SYSTEM      ")
print("================================================")
print("Type 'exit' to quit the application.\n")

while True:
    user_input = input("Enter your sentence: ")
    if user_input.lower() == 'exit':
        print("Goodbye!")
        break
        
    if not user_input.strip():
        continue

    cleaned_input = preprocess_text(user_input)
    input_tfidf = vectorizer.transform([cleaned_input])
    
    prediction = model.predict(input_tfidf)[0]
    probabilities = model.predict_proba(input_tfidf)[0]
    
    print(f"\nPredicted Sentiment: {prediction}")
    print("Prediction Probabilities:")
    for cls, prob in zip(model.classes_, probabilities):
        print(f"   - {cls}: {prob*100:.1f}%")
    print("-" * 50)