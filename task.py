import pandas as pd
import nltk
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Download stopwords (runs once)
nltk.download('stopwords')

# -----------------------------
# 1. SAMPLE TEXT DATA
# -----------------------------
data = {
    "text": [
        "I love this product",
        "This is the worst experience",
        "Amazing quality and great service",
        "Very bad and disappointing",
        "It is okay"
    ],
    "sentiment": [
        "positive",
        "negative",
        "positive",
        "negative",
        "neutral"
    ]
}

df = pd.DataFrame(data)

print("\n--- ORIGINAL DATA ---")
print(df)

# -----------------------------
# 2. TEXT CLEANING
# -----------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = " ".join(
        word for word in text.split()
        if word not in stopwords.words('english')
    )
    return text

df["clean_text"] = df["text"].apply(clean_text)

print("\n--- CLEANED TEXT ---")
print(df[["text", "clean_text"]])

# -----------------------------
# 3. VECTORIZATION
# -----------------------------
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df["clean_text"])
y = df["sentiment"]

print("\n--- VECTORIZED DATA ---")
print(X.toarray())
print("Features:", vectorizer.get_feature_names_out())

# -----------------------------
# 4. SENTIMENT MODEL
# -----------------------------
model = MultinomialNB()
model.fit(X, y)

print("\n--- MODEL TRAINED SUCCESSFULLY ---")

# -----------------------------
# 5. PREDICTION OUTPUT
# -----------------------------
test_sentence = ["This product is amazing"]
test_clean = [clean_text(test_sentence[0])]
test_vector = vectorizer.transform(test_clean)
prediction = model.predict(test_vector)

print("\n--- FINAL OUTPUT ---")
print("Sentence:", test_sentence[0])
print("Predicted Sentiment:", prediction[0])
