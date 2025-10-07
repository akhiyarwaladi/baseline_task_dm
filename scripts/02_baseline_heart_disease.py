"""
Baseline KNN - Heart Disease Prediction
Dataset: UCI Heart Disease (Cleveland) - 303 patients
Target: Binary classification (Healthy vs Disease)
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import sys
sys.path.append('..')
from utils import print_header, evaluate_classification
import warnings
warnings.filterwarnings('ignore')

print_header("BASELINE KNN - PREDIKSI PENYAKIT JANTUNG", level=1)

# Load data
columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
           'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']

df = pd.read_csv('../archive/dataset_heart_disease/processed.cleveland.data', names=columns, na_values='?')
df = df.dropna()
df['target'] = (df['target'] > 0).astype(int)

print_header("Dataset Info", level=2)
print(f"Total: {len(df)} patients")
print(f"Healthy: {(df['target']==0).sum()} | Disease: {(df['target']==1).sum()}")

# Split data
X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Preprocessing
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train KNN
print_header("K-NEAREST NEIGHBORS MODEL", level=1)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)

class_names = ['Healthy', 'Disease']
results = evaluate_classification(knn, X_train_scaled, X_test_scaled,
                                 y_train, y_test, class_names)

print_header("Summary", level=1)
print(f"Accuracy: {results['test_accuracy']:.4f}")
print(f"CV Score: {results['cv_mean']:.4f}")
