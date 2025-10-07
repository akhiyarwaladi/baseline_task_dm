"""
Baseline KNN - Tokopedia Review Emotion Classification
Dataset: PRDECT-ID (5,400 reviews)
Target: 5-class emotion (Happy, Sadness, Anger, Fear, Love)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
import sys
import os
# Add parent directory to path to import utils
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)
from utils import print_header, evaluate_classification, print_dataset_info
import warnings
warnings.filterwarnings('ignore')

print_header("BASELINE KNN - EMOTION CLASSIFICATION", level=1)

# Load data
df = pd.read_csv(os.path.join(parent_dir, 'dataset', '03_tokopedia_emotion.csv'))
df = df.dropna(subset=['Customer Review', 'Emotion'])
print_dataset_info(df, 'Emotion')

# Show examples
print_header("Review Examples per Emotion", level=2)
for emotion in ['Happy', 'Sadness', 'Anger', 'Fear', 'Love']:
    if emotion in df['Emotion'].values:
        sample = df[df['Emotion'] == emotion]['Customer Review'].iloc[0]
        print(f"\n{emotion}:")
        print(f'  "{sample[:100]}..."' if len(sample) > 100 else f'  "{sample}"')

# Preprocessing - TF-IDF
X = df['Customer Review']
y = df['Emotion']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print_header("TF-IDF Vectorization", level=2)
vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2), min_df=2)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)
print(f"Training shape: {X_train_tfidf.shape} | Testing shape: {X_test_tfidf.shape}")

# Train KNN
print_header("MODEL: K-NEAREST NEIGHBORS", level=1)
knn = KNeighborsClassifier(n_neighbors=7, metric='cosine', weights='distance')
knn.fit(X_train_tfidf, y_train)

class_names = sorted(y.unique())
results = evaluate_classification(knn, X_train_tfidf, X_test_tfidf,
                                 y_train, y_test, class_names)

# Top keywords per emotion
print_header("TOP KEYWORDS PER EMOTION", level=1)
feature_names = vectorizer.get_feature_names_out()
y_train_np = y_train.values

for emotion in class_names:
    emotion_tfidf = X_train_tfidf[y_train_np == emotion].mean(axis=0).A1
    top_indices = emotion_tfidf.argsort()[-10:][::-1]

    print_header(f"Top 10 for {emotion}", level=2)
    for i, idx in enumerate(top_indices, 1):
        print(f"{i:2d}. {feature_names[idx][:30].ljust(30)} : {emotion_tfidf[idx]:.4f}")

# Test with examples
print_header("PREDICTION EXAMPLES", level=1)

test_reviews = [
    ("Barangnya bagus banget! Sesuai ekspektasi, packing rapi. Sangat puas!", "Happy"),
    ("Kecewa berat, barang rusak pas datang. Tidak sesuai deskripsi.", "Sadness"),
    ("Pengiriman lama sekali! Seller tidak responsif. Sangat mengecewakan!", "Anger"),
    ("Barang jelek, takut rusak. Khawatir kualitasnya tidak tahan lama.", "Fear"),
    ("Perfect! Sangat suka dengan produk ini. Recommended seller!", "Love")
]

emoji_map = {'Happy': '😊', 'Sadness': '😢', 'Anger': '😠', 'Fear': '😨', 'Love': '❤️'}

for i, (review, expected) in enumerate(test_reviews, 1):
    review_tfidf = vectorizer.transform([review])
    pred = knn.predict(review_tfidf)[0]
    proba = knn.predict_proba(review_tfidf)[0]

    emoji = emoji_map.get(pred, '')
    confidence = proba[list(class_names).index(pred)]

    print(f"\n{i}. {emoji} {pred} ({confidence:.1%}) | Expected: {expected}")
    print(f'   "{review[:70]}..."')
    print(f"   Proba: {', '.join([f'{e}={p:.1%}' for e, p in zip(class_names, proba)])}")
