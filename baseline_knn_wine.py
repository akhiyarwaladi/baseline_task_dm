"""
Baseline KNN untuk Wine Recognition Dataset
Dataset: UCI Wine Dataset
Tutorial Data Mining

Dataset ini menggunakan analisis kimia dari wine untuk mengklasifikasikan
3 jenis cultivar yang berbeda dengan 13 atribut kimia.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings
warnings.filterwarnings('ignore')

# Load dataset
print("=" * 60)
print("BASELINE KNN - WINE RECOGNITION DATASET")
print("=" * 60)

# Definisi nama kolom (13 atribut kimia)
column_names = [
    'class',           # Target (1, 2, 3)
    'alcohol',
    'malic_acid',
    'ash',
    'alcalinity_of_ash',
    'magnesium',
    'total_phenols',
    'flavanoids',
    'nonflavanoid_phenols',
    'proanthocyanins',
    'color_intensity',
    'hue',
    'od280_od315',
    'proline'
]

# Load data
df = pd.read_csv('wine/wine.data', names=column_names, header=None)

print(f"\n1. DATA LOADING")
print(f"   - Total samples: {len(df)}")
print(f"   - Features: {len(column_names) - 1}")
print(f"   - Missing values: {df.isnull().sum().sum()}")

# Distribusi kelas
print(f"\n2. CLASS DISTRIBUTION")
for class_label in sorted(df['class'].unique()):
    count = (df['class'] == class_label).sum()
    print(f"   - Class {class_label}: {count} samples ({count/len(df)*100:.1f}%)")

# Split features and target
X = df.drop('class', axis=1)
y = df['class']

print(f"\n3. FEATURES ({X.shape[1]} attributes)")
for i, col in enumerate(X.columns, 1):
    print(f"   {i:2d}. {col:22s} - range: [{X[col].min():.2f}, {X[col].max():.2f}]")

# Split train-test (80-20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n4. TRAIN-TEST SPLIT")
print(f"   - Training set: {len(X_train)} samples")
print(f"   - Test set: {len(X_test)} samples")

# Standardization (penting untuk KNN karena berbasis jarak)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\n5. PREPROCESSING")
print(f"   - Standardization: Applied (mean=0, std=1)")

# Train KNN dengan k=5 (baseline)
k = 5
knn = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
knn.fit(X_train_scaled, y_train)

print(f"\n6. MODEL TRAINING")
print(f"   - Algorithm: K-Nearest Neighbors")
print(f"   - K value: {k}")
print(f"   - Distance metric: Euclidean")

# Prediction
y_pred = knn.predict(X_test_scaled)

# Evaluation
print(f"\n7. EVALUATION RESULTS")
print(f"\n   Test Set Performance:")
accuracy = accuracy_score(y_test, y_pred)
print(f"   - Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

print(f"\n   Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(f"   {cm}")

print(f"\n   Classification Report:")
class_names = [f'Class {i}' for i in sorted(df['class'].unique())]
print(classification_report(y_test, y_pred, target_names=class_names))

# Cross-validation untuk validasi model
cv_scores = cross_val_score(knn, X_train_scaled, y_train, cv=5, scoring='accuracy')
print(f"\n   5-Fold Cross-Validation (on training set):")
print(f"   - Scores: {cv_scores}")
print(f"   - Mean CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# Eksperimen dengan berbagai nilai K
print(f"\n8. K-VALUE COMPARISON")
k_values = [1, 3, 5, 7, 9, 11, 15, 21]
k_scores = []

for k_val in k_values:
    knn_temp = KNeighborsClassifier(n_neighbors=k_val)
    knn_temp.fit(X_train_scaled, y_train)
    score = knn_temp.score(X_test_scaled, y_test)
    k_scores.append(score)
    print(f"   k={k_val:2d} -> Accuracy: {score:.4f}")

best_k = k_values[np.argmax(k_scores)]
best_score = max(k_scores)
print(f"\n   Best K: {best_k} with accuracy {best_score:.4f}")

print("\n" + "=" * 60)
print("BASELINE KNN TRAINING COMPLETED")
print("=" * 60)
