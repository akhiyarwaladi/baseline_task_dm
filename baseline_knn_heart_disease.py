"""
=============================================================================
BASELINE KNN - HEART DISEASE PREDICTION
=============================================================================
Dataset: UCI Heart Disease (Cleveland)
Tujuan: Memprediksi apakah pasien memiliki penyakit jantung atau tidak

KONTEKS DUNIA NYATA:
Dataset berisi data medis pasien untuk prediksi penyakit jantung berdasarkan
hasil pemeriksaan kesehatan seperti tekanan darah, kolesterol, detak jantung, dll.

TARGET:
- 0 = SEHAT (Tidak ada penyakit jantung)
- 1 = SAKIT (Ada penyakit jantung)
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
print("BASELINE KNN - PREDIKSI PENYAKIT JANTUNG")
print("=" * 70)

# =============================================================================
# 1. LOAD DATA
# =============================================================================
print("\n[1] LOADING DATA...")

column_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
                'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']

df = pd.read_csv('heart+disease/processed.cleveland.data',
                 names=column_names, na_values='?')

print(f"    Total data: {len(df)} pasien")
print(f"    Missing values: {df.isnull().sum().sum()}")

# Hapus missing values
df = df.dropna()
print(f"    Data setelah cleaning: {len(df)} pasien")

# Convert target: 0=sehat, 1=sakit
df['target'] = (df['target'] > 0).astype(int)

# =============================================================================
# 2. EXPLORASI DATA
# =============================================================================
print("\n[2] EXPLORASI DATA")

print(f"\n    TARGET (Yang diprediksi):")
print(f"    - Kelas 0 (SEHAT): {(df['target']==0).sum()} pasien")
print(f"    - Kelas 1 (SAKIT): {(df['target']==1).sum()} pasien")

print(f"\n    FITUR (13 atribut pemeriksaan):")
features_info = {
    'age': 'Umur (tahun)',
    'sex': 'Jenis kelamin',
    'cp': 'Tipe nyeri dada',
    'trestbps': 'Tekanan darah (mm Hg)',
    'chol': 'Kolesterol (mg/dl)',
    'fbs': 'Gula darah puasa',
    'restecg': 'Hasil EKG',
    'thalach': 'Detak jantung max',
    'exang': 'Angina saat exercise',
    'oldpeak': 'ST depression',
    'slope': 'Slope ST segment',
    'ca': 'Jumlah pembuluh darah',
    'thal': 'Thalassemia'
}

for i, (col, desc) in enumerate(features_info.items(), 1):
    min_val = df[col].min()
    max_val = df[col].max()
    print(f"    {i:2d}. {col:12s} = {desc:25s} [{min_val:.0f} - {max_val:.0f}]")

# =============================================================================
# 3. SPLIT DATA
# =============================================================================
print("\n[3] SPLIT DATA (80% training, 20% testing)")

X = df.drop('target', axis=1)  # Features
y = df['target']                # Target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"    Training set: {len(X_train)} pasien")
print(f"    Testing set:  {len(X_test)} pasien")

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
print(f"    Penjelasan:")
print(f"    - True Negative (Sehat diprediksi Sehat): {cm[0,0]}")
print(f"    - False Positive (Sehat diprediksi Sakit): {cm[0,1]}")
print(f"    - False Negative (Sakit diprediksi Sehat): {cm[1,0]}")
print(f"    - True Positive (Sakit diprediksi Sakit): {cm[1,1]}")

# Classification Report
print(f"\n    Classification Report:")
print(classification_report(y_test, y_pred,
                          target_names=['Kelas 0: SEHAT', 'Kelas 1: SAKIT']))

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
# 8. PREDIKSI DATA BARU (SIMULASI)
# =============================================================================
print("\n" + "=" * 70)
print("[8] PREDIKSI PASIEN BARU - SIMULASI KASUS NYATA")
print("=" * 70)

# Data pasien baru yang ingin diprediksi
pasien_baru = pd.DataFrame([[45, 1, 2, 130, 233, 0, 0, 155, 0, 1.0, 1, 0, 3]],
                           columns=X.columns)

print("\n    Profil Pasien Baru:")
print(f"    - Umur: 45 tahun")
print(f"    - Jenis Kelamin: Pria")
print(f"    - Tekanan Darah: 130 mm Hg")
print(f"    - Kolesterol: 233 mg/dl")
print(f"    - Detak Jantung Max: 155")
print(f"    - Gula Darah Puasa: Normal")

# Standardisasi dan prediksi
pasien_scaled = scaler.transform(pasien_baru)
prediksi = knn.predict(pasien_scaled)[0]
probabilitas = knn.predict_proba(pasien_scaled)[0]

# Hasil
hasil = "SAKIT (Ada penyakit jantung)" if prediksi == 1 else "SEHAT (Tidak ada penyakit)"
print(f"\n    ╔════════════════════════════════════════════╗")
print(f"    ║  HASIL PREDIKSI: {hasil:23s} ║")
print(f"    ╚════════════════════════════════════════════╝")
print(f"\n    Probabilitas:")
print(f"    - Sehat (Kelas 0): {probabilitas[0]:.2%}")
print(f"    - Sakit (Kelas 1): {probabilitas[1]:.2%}")

print("\n    ⚠️  CATATAN: Ini hanya model pembelajaran, bukan diagnosis medis!")

print("\n" + "=" * 70)
print("SELESAI")
print("=" * 70)
