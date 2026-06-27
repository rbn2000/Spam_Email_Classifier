import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from preprocess import clean_text


# ---------------------------------
# Load Dataset
# ---------------------------------

df = pd.read_csv("spam.csv", encoding="latin-1")


# ---------------------------------
# Remove Unnecessary Columns
# ---------------------------------

df = df[['v1', 'v2']]

df.columns = ['label', 'message']


# ---------------------------------
# Encode Labels
# ham = 0
# spam = 1
# ---------------------------------

df['label'] = df['label'].map({
    'ham': 0,
    'spam': 1
})


# ---------------------------------
# Remove Duplicates
# ---------------------------------

df.drop_duplicates(inplace=True)


# ---------------------------------
# Clean Messages
# ---------------------------------

df['clean_message'] = df['message'].apply(clean_text)


# ---------------------------------
# Features & Labels
# ---------------------------------

X = df['clean_message']

y = df['label']


# ---------------------------------
# TF-IDF Vectorization
# ---------------------------------

vectorizer = TfidfVectorizer(max_features=5000)

X = vectorizer.fit_transform(X)


# ---------------------------------
# Train-Test Split
# ---------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# ---------------------------------
# Train Model
# ---------------------------------

model = MultinomialNB()

model.fit(X_train, y_train)


# ---------------------------------
# Prediction
# ---------------------------------

y_pred = model.predict(X_test)


# ---------------------------------
# Evaluation
# ---------------------------------

accuracy = accuracy_score(y_test, y_pred)

print("=" * 50)
print("Accuracy :", round(accuracy * 100, 2), "%")
print("=" * 50)

print("\nClassification Report\n")

print(classification_report(y_test, y_pred))

print("\nConfusion Matrix\n")

print(confusion_matrix(y_test, y_pred))


# ---------------------------------
# Save Model
# ---------------------------------

pickle.dump(model, open("model.pkl", "wb"))

pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("\nModel Saved Successfully!")