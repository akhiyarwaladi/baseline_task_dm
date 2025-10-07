"""
Baseline KNN - E-Commerce Customer Churn Prediction
Dataset: 5,630 customers with 20 features
Target: Binary classification (Churn / No Churn)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.metrics import roc_auc_score
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

print_header("BASELINE KNN - CUSTOMER CHURN PREDICTION", level=1)

# Load data
dataset_path = os.path.join(parent_dir, 'dataset', '04_ecommerce_churn.csv')
df = pd.read_csv(dataset_path)
print_dataset_info(df, 'Churn')

# Preprocessing
print_header("Data Preprocessing", level=2)

# Select relevant features
features_to_keep = [
    'Tenure', 'PreferredLoginDevice', 'CityTier', 'WarehouseToHome',
    'PreferredPaymentMode', 'Gender', 'HourSpendOnApp', 'NumberOfDeviceRegistered',
    'PreferedOrderCat', 'SatisfactionScore', 'MaritalStatus', 'NumberOfAddress',
    'Complain', 'OrderAmountHikeFromlastYear', 'CouponUsed', 'OrderCount',
    'DaySinceLastOrder', 'CashbackAmount', 'Churn'
]
df = df[features_to_keep]

# Handle missing values
df = df.dropna()

# Encode categorical features
categorical_cols = df.select_dtypes(include=['object']).columns
label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

print(f"Encoded {len(categorical_cols)} categorical features")
print(f"Final samples: {len(df):,}")

# Split data
X = df.drop('Churn', axis=1)
y = df['Churn']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Feature scaling (important for KNN!)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Training: {len(X_train):,} | Testing: {len(X_test):,}")

# ============================================================================
# MODEL 1: K-NEAREST NEIGHBORS
# ============================================================================
print_header("MODEL 1: K-NEAREST NEIGHBORS", level=1)
knn = KNeighborsClassifier(n_neighbors=9, metric='euclidean', weights='distance')
knn.fit(X_train_scaled, y_train)

class_names = ['No Churn', 'Churn']
knn_results = evaluate_classification(knn, X_train_scaled, X_test_scaled,
                                      y_train, y_test, class_names)

# ROC-AUC Score (important for imbalanced data)
y_pred_proba = knn.predict_proba(X_test_scaled)[:, 1]
roc_auc = roc_auc_score(y_test, y_pred_proba)

print_header("ROC-AUC Score", level=2)
print(f"ROC-AUC: {roc_auc:.4f}")

# ============================================================================
# MODEL 2: DECISION TREE
# ============================================================================
print_header("MODEL 2: DECISION TREE", level=1)

dt = DecisionTreeClassifier(
    max_depth=4,
    min_samples_split=50,
    min_samples_leaf=20,
    random_state=42
)
dt.fit(X_train, y_train)  # Use unscaled data for tree

dt_results = evaluate_classification(dt, X_train, X_test, y_train, y_test, class_names)

# Feature importance from Decision Tree
print_header("Top 10 Important Features (from Decision Tree)", level=2)
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': dt.feature_importances_
}).sort_values('importance', ascending=False).head(10)

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
plt.title('Decision Tree - Customer Churn Prediction (Max Depth = 4)',
         fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()

output_file = 'decision_tree_churn.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
print(f"Saved: {output_file} (300 DPI)")

# ============================================================================
# MODEL COMPARISON
# ============================================================================
print_header("MODEL COMPARISON", level=1)

print(f"{'Model':<20} {'Accuracy':<12} {'CV Score':<12} {'ROC-AUC':<12}")
print(f"{'-'*56}")
print(f"{'KNN (K=9)':<20} {knn_results['test_accuracy']:.4f}       {knn_results['cv_mean']:.4f}       {roc_auc:.4f}")
print(f"{'Decision Tree':<20} {dt_results['test_accuracy']:.4f}       {dt_results['cv_mean']:.4f}       N/A")

best_model = "KNN" if knn_results['test_accuracy'] > dt_results['test_accuracy'] else "Decision Tree"
print(f"\nBest Model (Accuracy): {best_model}")
print(f"Best Model (ROC-AUC): KNN ({roc_auc:.4f})")

# Prediction examples
print_header("CHURN RISK CLASSIFICATION", level=1)

# Get some test samples
test_indices = [0, 10, 50, 100, 200]
for idx in test_indices:
    if idx < len(X_test):
        sample = X_test_scaled[idx:idx+1]
        pred = knn.predict(sample)[0]
        proba = knn.predict_proba(sample)[0]
        churn_prob = proba[1]

        risk = "🚨 HIGH RISK" if churn_prob > 0.7 else "⚠️  MEDIUM RISK" if churn_prob > 0.4 else "✅ LOW RISK"
        actual = "Churned" if y_test.iloc[idx] == 1 else "Not Churned"

        print(f"\nCustomer {idx+1}: {risk}")
        print(f"  Predicted: {'Churn' if pred == 1 else 'No Churn'} ({churn_prob:.1%}) | Actual: {actual}")
        print(f"  Tenure: {X_test.iloc[idx]['Tenure']:.0f} | "
              f"Complain: {X_test.iloc[idx]['Complain']:.0f} | "
              f"Satisfaction: {X_test.iloc[idx]['SatisfactionScore']:.0f}")

# ============================================================================
# SAVE MODELS
# ============================================================================
print_header("SAVE MODELS TO FILE", level=1)

model_path = os.path.join(parent_dir, 'models', '04_model_churn.pkl')
os.makedirs(os.path.join(parent_dir, 'models'), exist_ok=True)

with open(model_path, 'wb') as f:
    pickle.dump({
        'model': knn,
        'dt_model': dt,
        'scaler': scaler,
        'encoders': label_encoders,
        'columns': X.columns,
        'feature_names': list(X.columns)
    }, f)

print(f"✅ Saved models and preprocessors to: {model_path}")
print(f"   - KNN (K=9, euclidean, distance weighted): Accuracy {knn_results['test_accuracy']:.4f}, ROC-AUC {roc_auc:.4f}")
print(f"   - Decision Tree (depth=4): Accuracy {dt_results['test_accuracy']:.4f}")
print(f"   - StandardScaler + {len(label_encoders)} LabelEncoders")
print(f"   - Model size: {os.path.getsize(model_path) / 1024:.1f} KB")

