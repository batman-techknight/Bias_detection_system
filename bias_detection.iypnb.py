import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re

nltk.download('stopwords')

# Load dataset (100-line bias dataset you downloaded)
df = pd.read_csv("sample_bias_dataset.csv")

# Preprocessing function
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()
def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-z\\s]', '', text)
    words = text.split()
    words = [stemmer.stem(word) for word in words if word not in stop_words]
    return " ".join(words)

df['clean_text'] = df['text'].apply(preprocess)

# Features and labels
X = df['clean_text']
y = df['label']

# Vectorization
vectorizer = TfidfVectorizer()
X_vec = vectorizer.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.3, random_state=42)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=["Not Biased", "Biased"], yticklabels=["Not Biased", "Biased"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Test custom inputs
def check_bias(text):
    text = preprocess(text)
    vec = vectorizer.transform([text])
    pred = model.predict(vec)[0]
    return "Biased" if pred==1 else "Not Biased"

print(check_bias("Girls cannot be scientists."))
print(check_bias("Technology makes life easier."))