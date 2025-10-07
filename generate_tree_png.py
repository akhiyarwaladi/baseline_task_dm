"""
Script untuk generate visualisasi Decision Tree menggunakan matplotlib
"""

import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("GENERATING DECISION TREE VISUALIZATION")
print("=" * 70)

# Load data (simplified version)
print("\n[1] Loading data...")
column_names = [
    'age', 'workclass', 'fnlwgt', 'education', 'education-num',
    'marital-status', 'occupation', 'relationship', 'race', 'sex',
    'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income'
]

df_train = pd.read_csv('adult/adult.data', names=column_names,
                       skipinitialspace=True, na_values='?')
df_test = pd.read_csv('adult/adult.test', names=column_names,
                      skipinitialspace=True, na_values='?', skiprows=1)

df_test['income'] = df_test['income'].str.rstrip('.')
df = pd.concat([df_train, df_test], ignore_index=True)
df = df.dropna()

# Encode - GUNAKAN ONE-HOT ENCODING untuk Decision Tree!
categorical_cols = ['workclass', 'education', 'marital-status', 'occupation',
                   'relationship', 'race', 'sex', 'native-country']

# Target encoding (binary)
df['income_encoded'] = (df['income'] == '>50K').astype(int)

# One-Hot Encoding untuk fitur kategorikal
print("[2] One-Hot Encoding kategorikal...")
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True, dtype=int)

X = df_encoded.drop(['income', 'income_encoded'], axis=1)
y = df_encoded['income_encoded']

print(f"    Fitur setelah One-Hot Encoding: {X.shape[1]} kolom")

# Sampling
X_subset, _, y_subset, _ = train_test_split(X, y, train_size=0.1, random_state=42, stratify=y)
X_train, X_test, y_train, y_test = train_test_split(
    X_subset, y_subset, test_size=0.2, random_state=42, stratify=y_subset
)

print("[3] Training Decision Tree...")
dt = DecisionTreeClassifier(max_depth=3, random_state=42, min_samples_split=50)
dt.fit(X_train, y_train)

# Feature names - gunakan nama kolom asli dari One-Hot Encoding
feature_names = list(X.columns)

print("[4] Creating visualization...")

# Create figure dengan ukuran besar
fig, ax = plt.subplots(figsize=(25, 15))

# Plot tree
plot_tree(dt,
          feature_names=feature_names,
          class_names=['Income ≤50K', 'Income >50K'],
          filled=True,
          rounded=True,
          fontsize=10,
          ax=ax,
          impurity=False,
          proportion=True)

# Title
plt.title('Decision Tree - Income Prediction\n(Max Depth = 3)',
          fontsize=16, fontweight='bold', pad=20)

# Tight layout
plt.tight_layout()

# Save
output_file = 'decision_tree_income.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
print(f"\n✓ PNG berhasil dibuat: {output_file}")
print(f"  Resolution: 300 DPI (high quality)")
print(f"  Size: ~25x15 inches")
print(f"\n  Buka file untuk melihat visualisasi Decision Tree!")

print("\n" + "=" * 70)
print("DONE!")
print("=" * 70)
