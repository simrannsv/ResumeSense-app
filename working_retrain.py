"""
Working Retrain - Converts sparse matrix to dense array
"""

import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle

print("="*60)
print("RETRAINING MODEL - WORKING VERSION")
print("="*60)

# 1. Load data
print("\n1. Loading data...")
df = pd.read_csv('UpdatedResumeDataSet.csv')
df = df.dropna()
print(f"   ✓ Loaded {len(df)} resumes")

# 2. Clean text
print("\n2. Cleaning text...")
def clean_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', str(text))
    return text.lower()

df['cleaned'] = df['Resume'].apply(clean_text)
print("   ✓ Done")

# 3. TF-IDF
print("\n3. Creating TF-IDF...")
tfidf = TfidfVectorizer(max_features=1000, stop_words='english')
X_sparse = tfidf.fit_transform(df['cleaned'])

# THE FIX: Convert sparse matrix to dense array
X = X_sparse.toarray()  # ← THIS IS THE KEY FIX!
y = df['Category'].values

print(f"   ✓ X shape: {X.shape}")
print(f"   ✓ y shape: {y.shape}")
print(f"   ✓ X type: {type(X)}")

# 4. Split
print("\n4. Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"   ✓ Train: {X_train.shape}")
print(f"   ✓ Test: {X_test.shape}")

# 5. Train
print("\n5. Training model...")
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)
print("   ✓ Training complete!")

# 6. Evaluate
print("\n6. Evaluating...")
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"   ✓ Accuracy: {accuracy*100:.2f}%")

# 7. Quick test
print("\n7. Testing on sample data...")
for i in range(3):
    sample = df['cleaned'].iloc[i]
    actual = df['Category'].iloc[i]
    
    # Transform and convert to dense
    vec = tfidf.transform([sample]).toarray()
    pred = model.predict(vec)[0]
    conf = model.predict_proba(vec)[0].max()
    
    match = "✓" if pred == actual else "✗"
    print(f"   {match} Actual: {actual:20s} Pred: {pred:20s} ({conf*100:.0f}%)")

# 8. Save
print("\n8. Saving model...")
pickle.dump(tfidf, open('tfidf_vectorizer.pkl', 'wb'))
pickle.dump(model, open('trained_model.pkl', 'wb'))
print("   ✓ Saved!")

print("\n" + "="*60)
print("SUCCESS!")
print("="*60)
print(f"Accuracy: {accuracy*100:.2f}%")
print("\nNow create test script...")

