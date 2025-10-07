"""
Baseline KNN - SMS Spam Detection (Text Classification)
Dataset: 1,143 SMS berbahasa Indonesia
Target: 3-class classification (Ham, Promosi, Spam)
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

print_header("BASELINE KNN - SMS SPAM DETECTION", level=1)

# Load data
df = pd.read_csv(os.path.join(parent_dir, 'dataset', '02_sms_spam.csv'))
print_dataset_info(df, 'label')

# Show examples
print_header("SMS Examples", level=2)
label_names = {0: 'Ham (Normal)', 1: 'Promosi', 2: 'Spam'}
for label, name in label_names.items():
    sample = df[df['label'] == label]['Teks'].iloc[0]
    print(f"{name}:")
    print(f'  "{sample[:80]}..."')

# Preprocessing - TF-IDF
X = df['Teks']
y = df['label']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print_header("TF-IDF Vectorization", level=2)
vectorizer = TfidfVectorizer(max_features=500, ngram_range=(1, 2))
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)
print(f"Training shape: {X_train_tfidf.shape} | Testing shape: {X_test_tfidf.shape}")

# Train KNN
print_header("MODEL: K-NEAREST NEIGHBORS", level=1)
knn = KNeighborsClassifier(n_neighbors=5, metric='cosine')
knn.fit(X_train_tfidf, y_train)

class_names = ['Ham (Normal)', 'Promosi (Iklan)', 'Spam (Phishing)']
results = evaluate_classification(knn, X_train_tfidf, X_test_tfidf,
                                 y_train, y_test, class_names)

# Top keywords per class
print_header("TOP KEYWORDS PER CLASS", level=1)
feature_names = vectorizer.get_feature_names_out()
y_train_np = y_train.values

for label, name in label_names.items():
    class_tfidf = X_train_tfidf[y_train_np == label].mean(axis=0).A1
    top_indices = class_tfidf.argsort()[-10:][::-1]

    print_header(f"Top 10 for {name}", level=2)
    for i, idx in enumerate(top_indices, 1):
        print(f"{i:2d}. {feature_names[idx][:30].ljust(30)} : {class_tfidf[idx]:.4f}")

# Test with examples
print_header("PREDICTION EXAMPLES", level=1)

test_sms = [
    ("PROMO SPESIAL! Diskon 50% untuk paket internet. Klik link sekarang!", "Promosi"),
    ("Halo kak, gimana kabarnya? Besok kita jadi ketemuan kan?", "Ham"),
    ("GRATIS pulsa 100rb! Tekan *123# sekarang. Buruan sebelum terlambat!", "Spam"),
    ("Oke deh, nanti aku transfer ya. Nomor rekening yang kemarin itu kan?", "Ham"),
    ("Paket Flash 3GB hanya 20rb. Aktifkan sekarang di *363#", "Promosi")
]

label_map = {0: 'HAM', 1: 'PROMOSI', 2: 'SPAM'}
emoji_map = {0: '✅', 1: '📢', 2: '🚨'}

for i, (sms, expected) in enumerate(test_sms, 1):
    sms_tfidf = vectorizer.transform([sms])
    pred = knn.predict(sms_tfidf)[0]
    proba = knn.predict_proba(sms_tfidf)[0]

    emoji = emoji_map[pred]
    label = label_map[pred]
    confidence = proba[pred]

    print(f"\n{i}. {emoji} {label} ({confidence:.1%}) | Expected: {expected}")
    print(f'   "{sms[:70]}..."')
    print(f"   Proba: Ham={proba[0]:.1%}, Promo={proba[1]:.1%}, Spam={proba[2]:.1%}")
