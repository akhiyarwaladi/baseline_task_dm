"""
=============================================================================
BASELINE KNN & DECISION TREE - DETEKSI STUNTING BALITA
=============================================================================
Dataset: Stunting Balita Detection (121K rows)
Tujuan: Klasifikasi status stunting balita Indonesia

KONTEKS:
Stunting adalah kondisi gagal tumbuh pada balita akibat kekurangan gizi kronis.
Dataset ini berisi data antropometri balita Indonesia untuk mendeteksi stunting.
=============================================================================
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("BASELINE MODEL - DETEKSI STUNTING BALITA")
print("=" * 70)

# =============================================================================
# 1. LOAD DATA
# =============================================================================
print("\n[1] LOADING DATA...")

df = pd.read_csv('dataset/data_balita.csv')

print(f"    Total samples: {len(df):,} balita")
print(f"    Total features: {len(df.columns) - 1}")
print(f"    Missing values: {df.isnull().sum().sum()}")

# =============================================================================
# 2. EXPLORASI DATA
# =============================================================================
print("\n[2] EXPLORASI DATA")

print(f"\n    TARGET (Status Stunting):")
target_col = 'Status Gizi' if 'Status Gizi' in df.columns else 'status'
for status, count in df[target_col].value_counts().items():
    print(f"    - {status}: {count:,} balita ({count/len(df)*100:.1f}%)")

print(f"\n    FITUR:")
for i, col in enumerate(df.columns, 1):
    if col != target_col:
        print(f"    {i}. {col}")

# =============================================================================
# 3. PREPROCESSING
# =============================================================================
print("\n[3] PREPROCESSING")

# Pisahkan features dan target
X = df.drop(target_col, axis=1)
y = df[target_col]

# Handle categorical features jika ada
X = pd.get_dummies(X, drop_first=True)

print(f"    Total features setelah encoding: {X.shape[1]}")

# =============================================================================
# 4. SPLIT DATA
# =============================================================================
print("\n[4] SPLIT DATA (80% training, 20% testing)")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"    Training set: {len(X_train):,} samples")
print(f"    Testing set:  {len(X_test):,} samples")

# =============================================================================
# 5. BASELINE MODEL 1 - KNN
# =============================================================================
print("\n" + "=" * 70)
print("[5] BASELINE MODEL 1: K-NEAREST NEIGHBORS (KNN)")
print("=" * 70)

# Standardisasi (penting untuk KNN!)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Training
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)

print(f"\n    Training selesai (K=5)")

# Evaluasi
y_pred_knn = knn.predict(X_test_scaled)
acc_knn = accuracy_score(y_test, y_pred_knn)

print(f"\n    HASIL KNN:")
print(f"    Accuracy: {acc_knn:.4f} ({acc_knn*100:.2f}%)")
print(f"\n    Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_knn))
print(f"\n    Classification Report:")
print(classification_report(y_test, y_pred_knn))

# Cross-validation
cv_scores_knn = cross_val_score(knn, X_train_scaled, y_train, cv=5)
print(f"    5-Fold CV Accuracy: {cv_scores_knn.mean():.4f} (+/- {cv_scores_knn.std()*2:.4f})")

# =============================================================================
# 6. BASELINE MODEL 2 - DECISION TREE
# =============================================================================
print("\n" + "=" * 70)
print("[6] BASELINE MODEL 2: DECISION TREE")
print("=" * 70)

# Training (tidak perlu scaling untuk Decision Tree)
dt = DecisionTreeClassifier(random_state=42, max_depth=3)
dt.fit(X_train, y_train)

print(f"\n    Training selesai (max_depth=3)")

# Evaluasi
y_pred_dt = dt.predict(X_test)
acc_dt = accuracy_score(y_test, y_pred_dt)

print(f"\n    HASIL DECISION TREE:")
print(f"    Accuracy: {acc_dt:.4f} ({acc_dt*100:.2f}%)")
print(f"\n    Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_dt))
print(f"\n    Classification Report:")
print(classification_report(y_test, y_pred_dt))

# Cross-validation
cv_scores_dt = cross_val_score(dt, X_train, y_train, cv=5)
print(f"    5-Fold CV Accuracy: {cv_scores_dt.mean():.4f} (+/- {cv_scores_dt.std()*2:.4f})")

# Feature importance
print(f"\n    Top 5 Fitur Penting:")
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': dt.feature_importances_
}).sort_values('importance', ascending=False)

for i, row in feature_importance.head(5).iterrows():
    print(f"    {row['feature']:25s}: {row['importance']:.4f}")

# =============================================================================
# 7. VISUALISASI DECISION TREE - TEXT FORMAT
# =============================================================================
print("\n" + "=" * 70)
print("[7] VISUALISASI DECISION TREE - TEXT FORMAT")
print("=" * 70)

# Tampilkan urutan kelas
class_order = sorted(y.unique())
print("\n    URUTAN KELAS (sesuai posisi di array weights):")
print("    " + "=" * 66)
for idx, class_name in enumerate(class_order):
    class_count = (y == class_name).sum()
    print(f"    [{idx}] = {class_name:20s} ({class_count:,} balita)")
print("    " + "=" * 66)

# Export tree sebagai text
tree_rules = export_text(dt, feature_names=list(X.columns), show_weights=True)

print("\n    DECISION TREE RULES:")
print("    " + "=" * 66)
print()

# Print tree rules dengan penjelasan tambahan
for line in tree_rules.split('\n'):
    if line.strip():
        # Tambahkan penjelasan untuk baris weights
        if 'weights:' in line and 'class:' in line:
            # Extract class name
            class_name = line.split('class:')[1].strip()
            # Highlight dengan format yang lebih jelas
            base_line = line.split('class:')[0] + f'class: {class_name}'
            print(f"    {base_line}")

            # Extract weights
            weights_str = line.split('weights:')[1].split('class:')[0].strip()
            weights = eval(weights_str)  # Parse array

            # Tampilkan breakdown per kelas
            breakdown = " → "
            for idx, (w, cls) in enumerate(zip(weights, class_order)):
                if w > 0:
                    breakdown += f"{cls}={int(w)}, "
            breakdown = breakdown.rstrip(", ")
            print(f"    {' ' * (len(line.split('|---')[0]) + 4)}{breakdown}")
        else:
            print(f"    {line}")

print()
print("    " + "=" * 66)
print("    📌 Cara Membaca:")
print("    - <= : kurang dari atau sama dengan")
print("    - >  : lebih dari")
print("    - class: prediksi kelas stunting")
print("    - weights: [posisi_0, posisi_1, posisi_2, posisi_3]")
print("    - Angka di breakdown menunjukkan jumlah balita per kelas di node tersebut")
print("    - Class yang dipilih = kelas dengan jumlah terbanyak di node")

# =============================================================================
# 8. VISUALISASI DECISION TREE - PNG
# =============================================================================
print("\n" + "=" * 70)
print("[8] VISUALISASI DECISION TREE - PNG")
print("=" * 70)

print("\n    Membuat visualisasi pohon keputusan...")

# Create figure
fig, ax = plt.subplots(figsize=(20, 12))

# Plot tree
plot_tree(dt,
          feature_names=list(X.columns),
          class_names=sorted(y.unique()),
          filled=True,
          rounded=True,
          fontsize=10,
          ax=ax,
          impurity=False,
          proportion=True)

plt.title('Decision Tree - Deteksi Stunting Balita\n(Max Depth = 3)',
          fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()

# Save
output_file = 'decision_tree_stunting.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
print(f"\n    ✓ Visualisasi disimpan: {output_file}")
print(f"    Resolution: 300 DPI")

# =============================================================================
# 9. PERBANDINGAN MODEL
# =============================================================================
print("\n" + "=" * 70)
print("[9] PERBANDINGAN MODEL")
print("=" * 70)

print(f"\n    {'Model':<20s} {'Accuracy':<12s} {'CV Score':<12s}")
print(f"    {'-'*44}")
print(f"    {'KNN (K=5)':<20s} {acc_knn:.4f}       {cv_scores_knn.mean():.4f}")
print(f"    {'Decision Tree':<20s} {acc_dt:.4f}       {cv_scores_dt.mean():.4f}")

if acc_knn > acc_dt:
    print(f"\n    ✓ KNN memberikan hasil terbaik!")
elif acc_dt > acc_knn:
    print(f"\n    ✓ Decision Tree memberikan hasil terbaik!")
else:
    print(f"\n    ✓ Kedua model memberikan hasil yang sama!")

print("\n" + "=" * 70)
print("SELESAI")
print("=" * 70)
