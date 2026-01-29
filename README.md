

# Sentiment Analysis Using Naive Bayes (Python)

##  Project Overview

This project implements a **basic sentiment analysis system** using **Natural Language Processing (NLP)** and **Machine Learning** in Python.
It classifies text into **positive, negative, or neutral sentiment** using a **Naive Bayes classifier**.

The project demonstrates the **end-to-end NLP workflow**, including text cleaning, feature extraction, model training, and prediction.

---

##  Technologies & Libraries Used

* **Python**
* **Pandas** – Data handling
* **NLTK** – Text preprocessing and stopwords removal
* **Regular Expressions (re)** – Text cleaning
* **Scikit-learn**

  * `CountVectorizer` – Text vectorization
  * `MultinomialNB` – Naive Bayes classifier

---

##  Dataset Description

This project uses a **sample in-code dataset** for demonstration purposes.

### Dataset Structure:

| Column      | Description                                         |
| ----------- | --------------------------------------------------- |
| `text`      | Input sentence                                      |
| `sentiment` | Sentiment label (`positive`, `negative`, `neutral`) |

Example sentences include:

* “I love this product”
* “This is the worst experience”
* “It is okay”

---

##  Workflow Explanation (Complete Code Analysis)

### 1️⃣ Import Required Libraries

The script imports libraries for:

* Data manipulation
* Text preprocessing
* Feature extraction
* Machine learning model training

---

### 2️⃣ Download Stopwords

```python
nltk.download('stopwords')
```

Downloads English stopwords used for text cleaning.
This runs only once and is cached by NLTK.

---

### 3️⃣ Create Sample Dataset

```python
pd.DataFrame(data)
```

Creates a Pandas DataFrame containing text samples and their corresponding sentiment labels.

---

### 4️⃣ Text Cleaning & Preprocessing

A custom `clean_text()` function performs:

* Conversion to lowercase
* Removal of punctuation and special characters
* Removal of English stopwords

This step ensures cleaner and more meaningful input for the model.

---

### 5️⃣ Feature Extraction (Vectorization)

```python
CountVectorizer()
```

* Converts cleaned text into a **Bag-of-Words** representation
* Transforms text into numerical feature vectors
* Extracts vocabulary features used by the model

---

### 6️⃣ Model Training

```python
MultinomialNB()
```

* Trains a **Naive Bayes classifier** on the vectorized text data
* Learns word patterns associated with each sentiment class

---

### 7️⃣ Prediction on New Text

```python
"This product is amazing"
```

* Cleans the input sentence
* Converts it into vector form
* Predicts sentiment using the trained model

---

## 📊 Output Generated

The program prints:

* Original dataset
* Cleaned text output
* Vectorized numerical data
* Feature names
* Final predicted sentiment for the test sentence

Example output:

```
Sentence: This product is amazing
Predicted Sentiment: positive
```

---

## ▶️ How to Run the Project

### Prerequisites

Install required libraries:

```bash
pip install pandas nltk scikit-learn
```

### Run the Script

```bash
python task.py
```

---

##  Use Cases

* Introductory NLP projects
* Sentiment analysis learning
* Text classification demonstrations
* Academic mini-projects
* Machine learning practice





