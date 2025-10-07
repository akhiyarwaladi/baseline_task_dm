"""
Baseline KNN & Decision Tree - Deteksi Stunting Balita
Dataset: 121K balita Indonesia dengan data antropometri
Target: Klasifikasi status stunting (Normal, Stunted, Severely Stunted, Tinggi)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import pickle
import sys
import os
# Add parent directory to path to import utils
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)
from utils import print_header, evaluate_classification, print_dataset_info
import warnings
warnings.filterwarnings('ignore')

print_header("BASELINE MODEL - DETEKSI STUNTING BALITA", level=1)

# Load data
df = pd.read_csv(os.path.join(parent_dir, 'dataset', '01_stunting_balita.csv'))
target_col = 'Status Gizi' if 'Status Gizi' in df.columns else 'status'
print_dataset_info(df, target_col)

# Preprocessing
X = pd.get_dummies(df.drop(target_col, axis=1), drop_first=True)
y = df[target_col]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print_header("Data Split", level=2)
print(f"Training: {len(X_train):,} | Testing: {len(X_test):,}")

# ============================================================================
# MODEL 1: K-NEAREST NEIGHBORS
# ============================================================================
print_header("MODEL 1: K-NEAREST NEIGHBORS", level=1)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)

class_names = sorted(y.unique())
knn_results = evaluate_classification(knn, X_train_scaled, X_test_scaled,
                                     y_train, y_test, class_names)

# ============================================================================
# MODEL 2: DECISION TREE
# ============================================================================
print_header("MODEL 2: DECISION TREE", level=1)

dt = DecisionTreeClassifier(random_state=42, max_depth=3)
dt.fit(X_train, y_train)

dt_results = evaluate_classification(dt, X_train, X_test, y_train, y_test, class_names)

# Feature importance
print_header("Top 5 Important Features", level=2)
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': dt.feature_importances_
}).sort_values('importance', ascending=False).head(5)

for _, row in feature_importance.iterrows():
    print(f"{row['feature'][:30].ljust(30)} : {row['importance']:.4f}")

# ============================================================================
# DECISION TREE VISUALIZATION - TEXT
# ============================================================================
print_header("DECISION TREE VISUALIZATION - TEXT", level=1)

# Class mapping
print_header("Class Mapping", level=2)
for idx, class_name in enumerate(class_names):
    count = (y == class_name).sum()
    print(f"[{idx}] = {class_name[:25].ljust(25)} ({count:,} samples)")

# Tree rules
tree_rules = export_text(dt, feature_names=list(X.columns), show_weights=True)
print_header("Tree Rules", level=2)

for line in tree_rules.split('\n'):
    if line.strip():
        if 'weights:' in line and 'class:' in line:
            class_name = line.split('class:')[1].strip()
            base_line = line.split('class:')[0] + f'class: {class_name}'
            print(f"{base_line}")

            weights_str = line.split('weights:')[1].split('class:')[0].strip()
            weights = eval(weights_str)

            breakdown = " → " + ", ".join([f"{cls}={int(w)}" for w, cls in zip(weights, class_names) if w > 0])
            indent = len(line.split('|---')[0]) + 4 if '|---' in line else 4
            print(f"{' ' * indent}{breakdown}")
        else:
            print(f"{line}")

# ============================================================================
# DECISION TREE VISUALIZATION - PNG
# ============================================================================
print_header("DECISION TREE VISUALIZATION - PNG", level=1)

fig, ax = plt.subplots(figsize=(20, 12))
plot_tree(dt, feature_names=list(X.columns), class_names=class_names,
         filled=True, rounded=True, fontsize=10, ax=ax,
         impurity=False, proportion=True)
plt.title('Decision Tree - Deteksi Stunting Balita (Max Depth = 3)',
         fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()

output_file = 'decision_tree_stunting.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
print(f"Saved: {output_file} (300 DPI)")

# ============================================================================
# MODEL COMPARISON
# ============================================================================
print_header("MODEL COMPARISON", level=1)

print(f"{'Model':<20} {'Accuracy':<12} {'CV Score':<12}")
print(f"{'-'*44}")
print(f"{'KNN (K=5)':<20} {knn_results['test_accuracy']:.4f}       {knn_results['cv_mean']:.4f}")
print(f"{'Decision Tree':<20} {dt_results['test_accuracy']:.4f}       {dt_results['cv_mean']:.4f}")

best_model = "KNN" if knn_results['test_accuracy'] > dt_results['test_accuracy'] else "Decision Tree"
print(f"\nBest Model: {best_model}")

# ============================================================================
# SAVE MODELS
# ============================================================================
print_header("SAVE MODELS TO FILE", level=1)

model_path = os.path.join(parent_dir, 'models', '01_model_stunting.pkl')
os.makedirs(os.path.join(parent_dir, 'models'), exist_ok=True)

with open(model_path, 'wb') as f:
    pickle.dump({
        'model': knn,
        'dt_model': dt,
        'scaler': scaler,  # IMPORTANT: Save scaler for KNN!
        'columns': X.columns,
        'feature_names': list(X.columns)
    }, f)

print(f"✅ Saved both KNN and Decision Tree models to: {model_path}")
print(f"   - KNN (K=5): Accuracy {knn_results['test_accuracy']:.4f}")
print(f"   - Decision Tree (depth=3): Accuracy {dt_results['test_accuracy']:.4f}")
print(f"   - Model size: {os.path.getsize(model_path) / 1024:.1f} KB")
