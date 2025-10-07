"""
=============================================================================
BASELINE KNN - WINE CLASSIFICATION
=============================================================================
Dataset: UCI Wine Recognition Dataset
Tujuan: Mengklasifikasikan jenis wine (cultivar) berdasarkan analisis kimia

KONTEKS DUNIA NYATA:
Produsen wine menggunakan analisis kimia untuk quality control dan memastikan
keaslian jenis anggur (cultivar). Dataset ini berisi hasil analisis 13 komponen
kimia dari wine Italia untuk mengidentifikasi 3 jenis cultivar.

TARGET:
- Kelas 1 = Cultivar Barolo
- Kelas 2 = Cultivar Grignolino
- Kelas 3 = Cultivar Barbera
=============================================================================
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("BASELINE KNN - KLASIFIKASI JENIS WINE")
print("=" * 70)

# =============================================================================
# 1. LOAD DATA
# =============================================================================
print("\n[1] LOADING DATA...")

column_names = [
    'class', 'alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash',
    'magnesium', 'total_phenols', 'flavanoids', 'nonflavanoid_phenols',
    'proanthocyanins', 'color_intensity', 'hue', 'od280_od315', 'proline'
]

df = pd.read_csv('wine/wine.data', names=column_names, header=None)

print(f"    Total samples: {len(df)} wine")
print(f"    Total features: {len(column_names) - 1}")
print(f"    Missing values: {df.isnull().sum().sum()}")

# =============================================================================
# 2. EXPLORASI DATA
# =============================================================================
print("\n[2] EXPLORASI DATA")

print(f"\n    TARGET (Jenis Cultivar):")
cultivar_names = {1: 'Barolo', 2: 'Grignolino', 3: 'Barbera'}
for class_id in sorted(df['class'].unique()):
    count = (df['class'] == class_id).sum()
    cultivar = cultivar_names[class_id]
    print(f"    - Kelas {class_id} ({cultivar:12s}): {count:3d} samples ({count/len(df)*100:.1f}%)")

print(f"\n    FITUR (13 komponen hasil analisis kimia):")
features_info = {
    'alcohol': 'Kadar alkohol (%)',
    'malic_acid': 'Asam malat (g/L)',
    'ash': 'Abu (g/L)',
    'alcalinity_of_ash': 'Alkalinitas abu',
    'magnesium': 'Magnesium (mg/L)',
    'total_phenols': 'Total fenol',
    'flavanoids': 'Flavonoid',
    'nonflavanoid_phenols': 'Fenol non-flavonoid',
    'proanthocyanins': 'Proanthocyanin',
    'color_intensity': 'Intensitas warna',
    'hue': 'Hue (corak warna)',
    'od280_od315': 'Protein content',
    'proline': 'Prolin (mg/L)'
}

X = df.drop('class', axis=1)
for i, (col, desc) in enumerate(features_info.items(), 1):
    min_val = X[col].min()
    max_val = X[col].max()
    print(f"    {i:2d}. {col:22s} = {desc:25s} [{min_val:7.2f} - {max_val:7.2f}]")

# =============================================================================
# 3. SPLIT DATA
# =============================================================================
print("\n[3] SPLIT DATA (80% training, 20% testing)")

y = df['class']  # Target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"    Training set: {len(X_train)} samples")
print(f"    Testing set:  {len(X_test)} samples")

# =============================================================================
# 4. PREPROCESSING - STANDARDIZATION
# =============================================================================
print("\n[4] PREPROCESSING")
print("    Standardisasi fitur (penting untuk KNN!)")
print("    Mengubah semua fitur ke skala yang sama (mean=0, std=1)")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =============================================================================
# 5. TRAINING MODEL KNN
# =============================================================================
print("\n[5] TRAINING MODEL")

k = 5  # Jumlah tetangga terdekat
knn = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
knn.fit(X_train_scaled, y_train)

print(f"    Algoritma: K-Nearest Neighbors (KNN)")
print(f"    K = {k} (menggunakan 5 tetangga terdekat)")
print(f"    Distance metric: Euclidean")

# =============================================================================
# 6. EVALUASI MODEL
# =============================================================================
print("\n[6] EVALUASI MODEL")

# Prediksi
y_pred = knn.predict(X_test_scaled)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\n    Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

# Confusion Matrix
print(f"\n    Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(f"    {cm}")

# Classification Report
print(f"\n    Classification Report:")
class_names = ['Kelas 1: Barolo', 'Kelas 2: Grignolino', 'Kelas 3: Barbera']
print(classification_report(y_test, y_pred, target_names=class_names))

# Cross-Validation
cv_scores = cross_val_score(knn, X_train_scaled, y_train, cv=5)
print(f"    5-Fold Cross-Validation:")
print(f"    - CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")

# =============================================================================
# 7. EKSPERIMEN DENGAN BERBAGAI NILAI K
# =============================================================================
print("\n[7] EKSPERIMEN NILAI K")
print("    Mencoba berbagai nilai K untuk menemukan yang terbaik:\n")

k_values = [1, 3, 5, 7, 9, 11, 15, 21]
k_scores = []

for k_val in k_values:
    knn_temp = KNeighborsClassifier(n_neighbors=k_val)
    knn_temp.fit(X_train_scaled, y_train)
    score = knn_temp.score(X_test_scaled, y_test)
    k_scores.append(score)
    print(f"    K = {k_val:2d}  →  Accuracy = {score:.4f}")

best_k = k_values[np.argmax(k_scores)]
best_score = max(k_scores)
print(f"\n    ✓ Nilai K terbaik: {best_k} dengan accuracy {best_score:.4f}")

# =============================================================================
# 8. PREDIKSI SAMPLE BARU (SIMULASI)
# =============================================================================
print("\n" + "=" * 70)
print("[8] PREDIKSI WINE BARU - SIMULASI QUALITY CONTROL")
print("=" * 70)

# Sample wine baru dari hasil analisis laboratorium
wine_baru = pd.DataFrame([[13.5, 2.1, 2.4, 19.0, 110, 2.5, 2.8, 0.30,
                          1.9, 6.5, 0.95, 3.2, 890]],
                         columns=X.columns)

print("\n    Hasil Analisis Kimia Wine Baru:")
print(f"    - Alcohol: 13.5%")
print(f"    - Proline: 890 mg/L")
print(f"    - Total Phenols: 2.5")
print(f"    - Flavanoids: 2.8")
print(f"    - Color Intensity: 6.5")
print(f"    - Hue: 0.95")

# Standardisasi dan prediksi
wine_scaled = scaler.transform(wine_baru)
prediksi = knn.predict(wine_scaled)[0]
probabilitas = knn.predict_proba(wine_scaled)[0]

# Hasil
cultivar = cultivar_names[prediksi]
print(f"\n    ╔════════════════════════════════════════════╗")
print(f"    ║  HASIL: Kelas {prediksi} - Cultivar {cultivar:12s} ║")
print(f"    ╚════════════════════════════════════════════╝")
print(f"\n    Probabilitas setiap kelas:")
print(f"    - Barolo (Kelas 1):     {probabilitas[0]:.2%}")
print(f"    - Grignolino (Kelas 2): {probabilitas[1]:.2%}")
print(f"    - Barbera (Kelas 3):    {probabilitas[2]:.2%}")

print("\n    💡 Model membantu quality control dan verifikasi keaslian cultivar")

print("\n" + "=" * 70)
print("SELESAI")
print("=" * 70)
