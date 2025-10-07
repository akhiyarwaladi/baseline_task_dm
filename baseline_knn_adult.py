"""
=============================================================================
BASELINE KNN - INCOME PREDICTION
=============================================================================
Dataset: UCI Adult Census Income (1994)
Tujuan: Memprediksi apakah seseorang berpenghasilan >50K atau <=50K per tahun

KONTEKS DUNIA NYATA:
Dataset berisi data sensus penduduk Amerika Serikat tahun 1994. Digunakan untuk
memprediksi tingkat penghasilan berdasarkan informasi demografis dan pekerjaan.
Berguna untuk analisis ekonomi, perencanaan pajak, dan targeting marketing.

TARGET:
- Kelas 0 = Income <=50K (Penghasilan Rendah) - 75% populasi
- Kelas 1 = Income >50K (Penghasilan Tinggi) - 25% populasi
=============================================================================
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("BASELINE KNN - PREDIKSI INCOME")
print("=" * 70)

# =============================================================================
# 1. LOAD DATA
# =============================================================================
print("\n[1] LOADING DATA...")

column_names = [
    'age', 'workclass', 'fnlwgt', 'education', 'education-num',
    'marital-status', 'occupation', 'relationship', 'race', 'sex',
    'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income'
]

# Load training dan testing data
df_train = pd.read_csv('adult/adult.data', names=column_names,
                       skipinitialspace=True, na_values='?')
df_test = pd.read_csv('adult/adult.test', names=column_names,
                      skipinitialspace=True, na_values='?', skiprows=1)

# Gabungkan untuk preprocessing konsisten
df_test['income'] = df_test['income'].str.rstrip('.')  # Hapus titik di test set
df = pd.concat([df_train, df_test], ignore_index=True)

print(f"    Total data: {len(df)} orang")
print(f"    Missing values: {df.isnull().sum().sum()}")

# Hapus missing values
df = df.dropna()
print(f"    Data setelah cleaning: {len(df)} orang")

# =============================================================================
# 2. EXPLORASI DATA
# =============================================================================
print("\n[2] EXPLORASI DATA")

print(f"\n    TARGET (Tingkat Penghasilan):")
low = (df['income'] == '<=50K').sum()
high = (df['income'] == '>50K').sum()
print(f"    - Kelas 0 (Income <=50K): {low:5d} orang ({low/len(df)*100:.1f}%)")
print(f"    - Kelas 1 (Income >50K):  {high:5d} orang ({high/len(df)*100:.1f}%)")

# Kategorisasi fitur
numerical_cols = ['age', 'fnlwgt', 'education-num', 'capital-gain',
                  'capital-loss', 'hours-per-week']
categorical_cols = ['workclass', 'education', 'marital-status', 'occupation',
                   'relationship', 'race', 'sex', 'native-country']

print(f"\n    FITUR NUMERIK ({len(numerical_cols)} atribut):")
num_info = {
    'age': 'Umur (tahun)',
    'fnlwgt': 'Final weight',
    'education-num': 'Lama pendidikan (tahun)',
    'capital-gain': 'Keuntungan modal (USD)',
    'capital-loss': 'Kerugian modal (USD)',
    'hours-per-week': 'Jam kerja/minggu'
}
for i, col in enumerate(numerical_cols, 1):
    print(f"    {i}. {col:18s} = {num_info[col]}")

print(f"\n    FITUR KATEGORIKAL ({len(categorical_cols)} atribut):")
cat_info = {
    'workclass': 'Jenis pekerjaan',
    'education': 'Tingkat pendidikan',
    'marital-status': 'Status pernikahan',
    'occupation': 'Okupasi/profesi',
    'relationship': 'Hubungan keluarga',
    'race': 'Ras',
    'sex': 'Jenis kelamin',
    'native-country': 'Negara asal'
}
for i, col in enumerate(categorical_cols, 1):
    n_unique = df[col].nunique()
    print(f"    {i}. {col:18s} = {cat_info[col]:25s} ({n_unique} kategori)")

# =============================================================================
# 3. PREPROCESSING
# =============================================================================
print("\n[3] PREPROCESSING")

# Encode categorical variables
print("    Encoding fitur kategorikal menjadi angka...")
df_encoded = df.copy()
label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Encode target
le_target = LabelEncoder()
df_encoded['income'] = le_target.fit_transform(df_encoded['income'])

print("    ✓ Semua fitur kategorikal berhasil di-encode")

# Sampling 10% untuk efisiensi (KNN lambat pada dataset besar)
print("\n    Mengambil 10% sampel untuk efisiensi komputasi...")
print("    (KNN lambat pada dataset besar)")

X = df_encoded.drop('income', axis=1)
y = df_encoded['income']

X_subset, _, y_subset, _ = train_test_split(
    X, y, train_size=0.1, random_state=42, stratify=y
)
print(f"    Subset yang digunakan: {len(X_subset)} orang")

# =============================================================================
# 4. SPLIT DATA
# =============================================================================
print("\n[4] SPLIT DATA (80% training, 20% testing)")

X_train, X_test, y_train, y_test = train_test_split(
    X_subset, y_subset, test_size=0.2, random_state=42, stratify=y_subset
)

print(f"    Training set: {len(X_train)} orang")
print(f"    Testing set:  {len(X_test)} orang")

# Standardization
print("\n    Standardisasi fitur (penting untuk KNN!)")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =============================================================================
# 5. TRAINING MODEL KNN
# =============================================================================
print("\n[5] TRAINING MODEL")

k = 5  # Jumlah tetangga terdekat
knn = KNeighborsClassifier(n_neighbors=k, metric='euclidean', n_jobs=-1)
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
print(classification_report(y_test, y_pred,
                          target_names=['Kelas 0: <=50K', 'Kelas 1: >50K']))

# Cross-Validation (3-fold untuk efisiensi)
cv_scores = cross_val_score(knn, X_train_scaled, y_train, cv=3, n_jobs=-1)
print(f"    3-Fold Cross-Validation:")
print(f"    - CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")

# =============================================================================
# 7. EKSPERIMEN DENGAN BERBAGAI NILAI K
# =============================================================================
print("\n[7] EKSPERIMEN NILAI K")
print("    Mencoba berbagai nilai K untuk menemukan yang terbaik:\n")

k_values = [3, 5, 7, 9, 11]
k_scores = []

for k_val in k_values:
    knn_temp = KNeighborsClassifier(n_neighbors=k_val, n_jobs=-1)
    knn_temp.fit(X_train_scaled, y_train)
    score = knn_temp.score(X_test_scaled, y_test)
    k_scores.append(score)
    print(f"    K = {k_val:2d}  →  Accuracy = {score:.4f}")

best_k = k_values[np.argmax(k_scores)]
best_score = max(k_scores)
print(f"\n    ✓ Nilai K terbaik: {best_k} dengan accuracy {best_score:.4f}")

# =============================================================================
# 8. PREDIKSI PROFIL BARU (SIMULASI)
# =============================================================================
print("\n" + "=" * 70)
print("[8] PREDIKSI PROFIL BARU - SIMULASI ANALISIS DEMOGRAFI")
print("=" * 70)

# Profil orang baru (data raw sebelum encoding)
profil_raw = {
    'age': 45, 'workclass': 'Private', 'fnlwgt': 200000,
    'education': 'Masters', 'education-num': 14,
    'marital-status': 'Married-civ-spouse', 'occupation': 'Exec-managerial',
    'relationship': 'Husband', 'race': 'White', 'sex': 'Male',
    'capital-gain': 5000, 'capital-loss': 0, 'hours-per-week': 50,
    'native-country': 'United-States'
}

print("\n    Profil Orang Baru:")
print(f"    - Umur: {profil_raw['age']} tahun")
print(f"    - Jenis Kelamin: {profil_raw['sex']}")
print(f"    - Pendidikan: {profil_raw['education']} ({profil_raw['education-num']} tahun)")
print(f"    - Pekerjaan: {profil_raw['occupation']}")
print(f"    - Jam Kerja: {profil_raw['hours-per-week']} jam/minggu")
print(f"    - Status: {profil_raw['marital-status']}")
print(f"    - Capital Gain: ${profil_raw['capital-gain']}")

# Encode profil baru
profil_df = pd.DataFrame([profil_raw])
profil_encoded = profil_df.copy()

for col in categorical_cols:
    profil_encoded[col] = label_encoders[col].transform(profil_df[col])

profil_encoded = profil_encoded[X.columns]

# Standardisasi dan prediksi
profil_scaled = scaler.transform(profil_encoded)
prediksi = knn.predict(profil_scaled)[0]
probabilitas = knn.predict_proba(profil_scaled)[0]

# Hasil
hasil = "INCOME >50K (Tinggi)" if prediksi == 1 else "INCOME <=50K (Rendah)"
print(f"\n    ╔════════════════════════════════════════════╗")
print(f"    ║  HASIL PREDIKSI: {hasil:22s} ║")
print(f"    ╚════════════════════════════════════════════╝")
print(f"\n    Probabilitas:")
print(f"    - Income <=50K (Rendah): {probabilitas[0]:.2%}")
print(f"    - Income >50K (Tinggi):  {probabilitas[1]:.2%}")

print("\n    💡 Model berguna untuk analisis demografis dan perencanaan ekonomi")

print("\n" + "=" * 70)
print("SELESAI")
print(f"Catatan: Analisis ini menggunakan 10% subset data untuk efisiensi.")
print(f"Untuk dataset besar, pertimbangkan algoritma lain (Decision Tree, etc.)")
print("=" * 70)
