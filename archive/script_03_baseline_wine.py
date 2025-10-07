"""
Baseline KNN - Wine Classification
Dataset: UCI Wine Recognition - 178 samples
Target: 3-class classification (Barolo, Grignolino, Barbera)
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

print_header("BASELINE KNN - KLASIFIKASI JENIS WINE", level=1)

# Load data
columns = ['class', 'alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash',
           'magnesium', 'total_phenols', 'flavanoids', 'nonflavanoid_phenols',
           'proanthocyanins', 'color_intensity', 'hue', 'od280_od315', 'proline']

df = pd.read_csv('../archive/dataset_wine/wine.data', names=columns, header=None)

print_header("Dataset Info", level=2)
print(f"Total: {len(df)} wine samples")
print(f"Features: 13 chemical components")

cultivar_names = {1: 'Barolo', 2: 'Grignolino', 3: 'Barbera'}
for cls in sorted(df['class'].unique()):
    count = (df['class'] == cls).sum()
    print(f"Class {cls} ({cultivar_names[cls]}): {count} samples ({count/len(df)*100:.1f}%)")

# Split data
X = df.drop('class', axis=1)
y = df['class']

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

class_names = ['Barolo', 'Grignolino', 'Barbera']
results = evaluate_classification(knn, X_train_scaled, X_test_scaled,
                                 y_train, y_test, class_names)

print_header("Summary", level=1)
print(f"Accuracy: {results['test_accuracy']:.4f}")
print(f"CV Score: {results['cv_mean']:.4f}")
