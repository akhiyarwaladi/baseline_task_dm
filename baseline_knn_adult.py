"""
=============================================================================
BASELINE KNN & DECISION TREE - INCOME PREDICTION
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

ALGORITMA:
1. K-Nearest Neighbors (KNN) - Instance-based learning
2. Decision Tree - Rule-based learning (mudah diinterpretasi!)
=============================================================================
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, export_text, export_graphviz
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("BASELINE: KNN & DECISION TREE - PREDIKSI INCOME")
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

print("    📌 PENTING: Encoding Fitur Kategorikal")
print("    - KNN: Butuh Label Encoding (karena butuh angka untuk distance)")
print("    - Decision Tree: Butuh One-Hot Encoding (agar tidak misleading!)")
print()

# Label Encoding untuk TARGET (binary: 0/1)
df['income_encoded'] = (df['income'] == '>50K').astype(int)

# ONE-HOT ENCODING untuk Decision Tree (BENAR!)
print("    [Decision Tree] One-Hot Encoding kategorikal...")
print("    → Setiap kategori jadi kolom terpisah (0/1)")

df_onehot = pd.get_dummies(df, columns=categorical_cols, drop_first=True, dtype=int)
X_dt = df_onehot.drop(['income', 'income_encoded'], axis=1)
y = df['income_encoded']

print(f"    ✓ Fitur Decision Tree: {X_dt.shape[1]} kolom (dari {len(numerical_cols)} numerik + one-hot)")
print(f"      Contoh kolom: {list(X_dt.columns[:5])}...")

# LABEL ENCODING untuk KNN (untuk kompatibilitas)
print("\n    [KNN] Label Encoding kategorikal...")
print("    → Setiap kategori jadi angka (0, 1, 2, ...)")

df_encoded = df.copy()
label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df[col])
    label_encoders[col] = le

X_knn = df_encoded.drop(['income', 'income_encoded'], axis=1)

print(f"    ✓ Fitur KNN: {X_knn.shape[1]} kolom")
print()
print("    💡 One-Hot Encoding lebih baik untuk Decision Tree karena:")
print("       - Tidak ada asumsi urutan (Private ≠ lebih besar dari Government)")
print("       - Setiap kategori independen")
print("       - Interpretasi tree lebih jelas (misal: 'workclass_Self-emp-inc = 1')")

# Sampling 10% untuk efisiensi (KNN lambat pada dataset besar)
print("\n    Mengambil 10% sampel untuk efisiensi komputasi...")
print("    (KNN lambat pada dataset besar, Decision Tree lebih cepat)")

# Sampling untuk Decision Tree (One-Hot Encoding)
X_dt_subset, _, y_dt_subset, _ = train_test_split(
    X_dt, y, train_size=0.1, random_state=42, stratify=y
)

# Sampling untuk KNN (Label Encoding)
X_knn_subset, _, y_knn_subset, _ = train_test_split(
    X_knn, y, train_size=0.1, random_state=42, stratify=y
)
print(f"    Subset yang digunakan: {len(X_dt_subset)} orang")

# =============================================================================
# 4. SPLIT DATA
# =============================================================================
print("\n[4] SPLIT DATA (80% training, 20% testing)")

# Split untuk Decision Tree
X_dt_train, X_dt_test, y_dt_train, y_dt_test = train_test_split(
    X_dt_subset, y_dt_subset, test_size=0.2, random_state=42, stratify=y_dt_subset
)

# Split untuk KNN
X_knn_train, X_knn_test, y_knn_train, y_knn_test = train_test_split(
    X_knn_subset, y_knn_subset, test_size=0.2, random_state=42, stratify=y_knn_subset
)

print(f"    Training set: {len(X_dt_train)} orang")
print(f"    Testing set:  {len(X_dt_test)} orang")

# Standardization (hanya untuk KNN)
print("\n    Standardisasi fitur untuk KNN (tidak perlu untuk Decision Tree)")
scaler = StandardScaler()
X_knn_train_scaled = scaler.fit_transform(X_knn_train)
X_knn_test_scaled = scaler.transform(X_knn_test)

# =============================================================================
# 5. TRAINING MODEL KNN
# =============================================================================
print("\n[5] TRAINING MODEL - K-NEAREST NEIGHBORS (KNN)")

k = 5
knn = KNeighborsClassifier(n_neighbors=k, metric='euclidean', n_jobs=-1)
knn.fit(X_knn_train_scaled, y_knn_train)

print(f"    Algoritma: K-Nearest Neighbors")
print(f"    K = {k} tetangga terdekat")
print(f"    Distance metric: Euclidean")

# Evaluasi KNN
y_pred_knn = knn.predict(X_knn_test_scaled)
accuracy_knn = accuracy_score(y_knn_test, y_pred_knn)
print(f"\n    ✓ KNN Accuracy: {accuracy_knn:.4f} ({accuracy_knn*100:.2f}%)")

# =============================================================================
# 6. TRAINING MODEL DECISION TREE
# =============================================================================
print("\n[6] TRAINING MODEL - DECISION TREE")

# Limit depth agar tree mudah dibaca
dt = DecisionTreeClassifier(max_depth=5, random_state=42, min_samples_split=50)
dt.fit(X_dt_train, y_dt_train)  # Decision Tree tidak perlu scaling!

print(f"    Algoritma: Decision Tree (CART)")
print(f"    Max depth: 5 (dibatasi agar mudah dibaca)")
print(f"    Min samples split: 50")
print(f"    Criterion: Gini impurity")

# Evaluasi Decision Tree
y_pred_dt = dt.predict(X_dt_test)
accuracy_dt = accuracy_score(y_dt_test, y_pred_dt)
print(f"\n    ✓ Decision Tree Accuracy: {accuracy_dt:.4f} ({accuracy_dt*100:.2f}%)")

# =============================================================================
# 7. PERBANDINGAN MODEL
# =============================================================================
print("\n[7] PERBANDINGAN KNN vs DECISION TREE")

print(f"\n    {'Metric':<20} {'KNN':>15} {'Decision Tree':>15}")
print(f"    {'-'*20} {'-'*15} {'-'*15}")
print(f"    {'Accuracy':<20} {accuracy_knn:>15.4f} {accuracy_dt:>15.4f}")

# Confusion Matrix
cm_knn = confusion_matrix(y_knn_test, y_pred_knn)
cm_dt = confusion_matrix(y_dt_test, y_pred_dt)

print(f"\n    Confusion Matrix - KNN:")
print(f"    {cm_knn}")

print(f"\n    Confusion Matrix - Decision Tree:")
print(f"    {cm_dt}")

# Classification Report
print(f"\n    Classification Report - KNN:")
print(classification_report(y_knn_test, y_pred_knn,
                          target_names=['<=50K', '>50K'], digits=3))

print(f"\n    Classification Report - Decision Tree:")
print(classification_report(y_dt_test, y_pred_dt,
                          target_names=['<=50K', '>50K'], digits=3))

# =============================================================================
# 8. VISUALISASI DECISION TREE (TEXT FORMAT)
# =============================================================================
print("\n" + "=" * 70)
print("[8] VISUALISASI DECISION TREE - RULES YANG MUDAH DIBACA")
print("=" * 70)

# Export tree sebagai text
tree_rules = export_text(dt, feature_names=list(X_dt.columns),
                        max_depth=3, show_weights=True)

print("\n    DECISION TREE RULES (3 level pertama):")
print("    " + "=" * 66)
print()

# Parse dan format tree rules agar lebih readable
for line in tree_rules.split('\n')[:40]:  # Ambil 40 baris pertama
    if line.strip():
        print(f"    {line}")

print()
print("    " + "=" * 66)
print("    📌 Cara Membaca:")
print("    - <= : kurang dari atau sama dengan")
print("    - >  : lebih dari")
print("    - class: prediksi (0=<=50K, 1=>50K)")
print("    - weights: [jumlah <=50K, jumlah >50K] di node tersebut")
print()
print("    💡 One-Hot Encoding:")
print("    - Kolom seperti 'workclass_Private' = 1 artinya jenis pekerjaan adalah Private")
print("    - Kolom seperti 'sex_Male' = 1 artinya jenis kelamin adalah Male")

# =============================================================================
# 8B. EXPORT GRAPHVIZ (VISUALISASI GRAFIS)
# =============================================================================
print("\n" + "=" * 70)
print("[8B] EXPORT DECISION TREE KE GRAPHVIZ (Visualisasi Grafis)")
print("=" * 70)

# Export ke file .dot (gunakan nama kolom asli dari One-Hot Encoding)
dot_file = 'decision_tree_income.dot'
export_graphviz(
    dt,
    out_file=dot_file,
    feature_names=list(X_dt.columns),
    class_names=['Income_<=50K', 'Income_>50K'],
    filled=True,           # Warna berdasarkan class
    rounded=True,          # Node dengan sudut rounded
    special_characters=True,
    max_depth=3,           # Limit depth agar tidak terlalu kompleks
    impurity=False,        # Tidak tampilkan gini (agar lebih simple)
    proportion=True        # Tampilkan proporsi samples
)

print(f"\n    ✓ Decision Tree berhasil di-export ke: {dot_file}")
print(f"\n    📊 Cara Membuat Visualisasi Grafis:")
print(f"    ")
print(f"    1. Install Graphviz (jika belum):")
print(f"       Ubuntu/Debian: sudo apt-get install graphviz")
print(f"       MacOS:         brew install graphviz")
print(f"       Windows:       Download dari graphviz.org")
print(f"    ")
print(f"    2. Convert .dot ke PNG:")
print(f"       dot -Tpng {dot_file} -o decision_tree_income.png")
print(f"    ")
print(f"    3. Atau online: https://dreampuf.github.io/GraphvizOnline/")
print(f"       → Copy isi file {dot_file} ke website tersebut")

# Coba generate PNG jika graphviz tersedia
try:
    import subprocess
    result = subprocess.run(['dot', '-Tpng', dot_file, '-o', 'decision_tree_income.png'],
                          capture_output=True, timeout=10)
    if result.returncode == 0:
        print(f"\n    ✓ PNG berhasil dibuat: decision_tree_income.png")
        print(f"    Buka file PNG untuk melihat visualisasi tree!")
    else:
        print(f"\n    ℹ️  Graphviz belum terinstall. Gunakan cara manual di atas.")
except (FileNotFoundError, subprocess.TimeoutExpired):
    print(f"\n    ℹ️  Graphviz belum terinstall. Gunakan cara manual di atas.")

# Feature Importance
print("\n" + "=" * 70)
print("[9] FEATURE IMPORTANCE (Top 10)")
print("    Fitur mana yang paling penting untuk prediksi?\n")

importances = dt.feature_importances_
feature_importance = pd.DataFrame({
    'Feature': X_dt.columns,
    'Importance': importances
}).sort_values('Importance', ascending=False)

for idx, row in feature_importance.head(10).iterrows():
    feat = row['Feature']
    imp = row['Importance']
    bar = '█' * int(imp * 50)
    print(f"    {feat:30s} {bar} {imp:.4f}")

# =============================================================================
# 10. PREDIKSI DATA BARU DENGAN KEDUA MODEL
# =============================================================================
print("\n" + "=" * 70)
print("[10] PREDIKSI PROFIL BARU - PERBANDINGAN KNN vs DECISION TREE")
print("=" * 70)

# Profil orang baru
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

# Encode profil baru untuk KNN (Label Encoding)
profil_df = pd.DataFrame([profil_raw])
profil_knn = profil_df.copy()

for col in categorical_cols:
    profil_knn[col] = label_encoders[col].transform(profil_df[col])

profil_knn = profil_knn[X_knn.columns]

# Prediksi dengan KNN
profil_knn_scaled = scaler.transform(profil_knn)
prediksi_knn = knn.predict(profil_knn_scaled)[0]
prob_knn = knn.predict_proba(profil_knn_scaled)[0]

# Encode profil baru untuk Decision Tree (One-Hot Encoding)
profil_dt = pd.get_dummies(profil_df, columns=categorical_cols, drop_first=True, dtype=int)

# Pastikan semua kolom sama dengan X_dt
for col in X_dt.columns:
    if col not in profil_dt.columns:
        profil_dt[col] = 0

profil_dt = profil_dt[X_dt.columns]

# Prediksi dengan Decision Tree
prediksi_dt = dt.predict(profil_dt)[0]
prob_dt = dt.predict_proba(profil_dt)[0]

# Hasil
print("\n    " + "=" * 66)
print("    HASIL PREDIKSI:")
print("    " + "=" * 66)

hasil_knn = "INCOME >50K (Tinggi)" if prediksi_knn == 1 else "INCOME <=50K (Rendah)"
hasil_dt = "INCOME >50K (Tinggi)" if prediksi_dt == 1 else "INCOME <=50K (Rendah)"

print(f"\n    🔹 K-NEAREST NEIGHBORS (KNN):")
print(f"       Prediksi: {hasil_knn}")
print(f"       Probabilitas: <=50K={prob_knn[0]:.2%}, >50K={prob_knn[1]:.2%}")

print(f"\n    🔹 DECISION TREE:")
print(f"       Prediksi: {hasil_dt}")
print(f"       Probabilitas: <=50K={prob_dt[0]:.2%}, >50K={prob_dt[1]:.2%}")

# Decision Path untuk Decision Tree
print("\n    📋 DECISION PATH (Bagaimana Decision Tree memutuskan?):")
print("    " + "-" * 66)

decision_path = dt.decision_path(profil_dt)
node_indicator = decision_path.toarray()[0]
leaf_id = dt.apply(profil_dt)[0]

feature = dt.tree_.feature
threshold = dt.tree_.threshold

print()
for node_id in range(len(node_indicator)):
    if node_indicator[node_id]:
        if leaf_id == node_id:
            print(f"    → Sampai di LEAF NODE → Prediksi: {hasil_dt}")
        else:
            feat_name = X_dt.columns[feature[node_id]]
            feat_value = profil_dt[feat_name].values[0]
            thresh = threshold[node_id]

            if feat_value <= thresh:
                direction = f"<= {thresh:.2f}"
                print(f"    → {feat_name} = {feat_value:.2f} {direction} ✓")
            else:
                direction = f"> {thresh:.2f}"
                print(f"    → {feat_name} = {feat_value:.2f} {direction} ✓")

print("\n" + "=" * 70)
print("SELESAI")
print(f"\n💡 INSIGHTS:")
print(f"   - Decision Tree lebih mudah diinterpretasi (bisa lihat rules)")
print(f"   - KNN lebih flexible tapi sulit dijelaskan (black box)")
print(f"   - Decision Tree lebih cepat untuk dataset besar")
print(f"   - Pilih model berdasarkan trade-off interpretability vs accuracy")
print("=" * 70)
