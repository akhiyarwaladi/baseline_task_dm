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
from sklearn.metrics import roc_auc_score
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

# Train KNN
print_header("MODEL: K-NEAREST NEIGHBORS", level=1)
knn = KNeighborsClassifier(n_neighbors=9, metric='euclidean', weights='distance')
knn.fit(X_train_scaled, y_train)

class_names = ['No Churn', 'Churn']
results = evaluate_classification(knn, X_train_scaled, X_test_scaled,
                                 y_train, y_test, class_names)

# ROC-AUC Score (important for imbalanced data)
y_pred_proba = knn.predict_proba(X_test_scaled)[:, 1]
roc_auc = roc_auc_score(y_test, y_pred_proba)

print_header("ROC-AUC Score", level=2)
print(f"ROC-AUC: {roc_auc:.4f}")

# Feature importance analysis
print_header("TOP 10 IMPORTANT FEATURES", level=1)

# Calculate feature importance using correlation with target
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': np.abs([df[col].corr(y) for col in X.columns])
}).sort_values('importance', ascending=False).head(10)

for i, row in feature_importance.iterrows():
    print(f"{row['feature'][:30].ljust(30)} : {row['importance']:.4f}")

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

print_header("Model Summary", level=1)
print(f"Accuracy    : {results['test_accuracy']:.4f}")
print(f"ROC-AUC     : {roc_auc:.4f}")
print(f"CV Score    : {results['cv_mean']:.4f}")
