"""
Baseline KNN untuk Adult Income Dataset
Dataset: UCI Adult Census Income
Tutorial Data Mining

Dataset ini memprediksi apakah seseorang memiliki income >50K atau <=50K
berdasarkan data census (umur, pendidikan, pekerjaan, dll).
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings
warnings.filterwarnings('ignore')

# Load dataset
print("=" * 60)
print("BASELINE KNN - ADULT INCOME DATASET")
print("=" * 60)

# Definisi nama kolom
column_names = [
    'age', 'workclass', 'fnlwgt', 'education', 'education-num',
    'marital-status', 'occupation', 'relationship', 'race', 'sex',
    'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income'
]

# Load data training
df_train = pd.read_csv('adult/adult.data', names=column_names,
                       skipinitialspace=True, na_values='?')

# Load data test
df_test = pd.read_csv('adult/adult.test', names=column_names,
                      skipinitialspace=True, na_values='?', skiprows=1)

# Gabungkan untuk preprocessing yang konsisten
df_test['income'] = df_test['income'].str.rstrip('.')  # Hilangkan titik di test set
df = pd.concat([df_train, df_test], ignore_index=True)

print(f"\n1. DATA LOADING")
print(f"   - Training samples: {len(df_train)}")
print(f"   - Test samples: {len(df_test)}")
print(f"   - Total samples: {len(df)}")
print(f"   - Missing values: {df.isnull().sum().sum()}")

# Handle missing values
df_clean = df.dropna()
print(f"   - Samples after removing missing values: {len(df_clean)}")

# Distribusi target
print(f"\n2. TARGET DISTRIBUTION")
print(f"   - Income <=50K: {(df_clean['income'] == '<=50K').sum()} samples")
print(f"   - Income >50K: {(df_clean['income'] == '>50K').sum()} samples")

# Identifikasi kolom kategorikal dan numerikal
categorical_cols = ['workclass', 'education', 'marital-status', 'occupation',
                   'relationship', 'race', 'sex', 'native-country']
numerical_cols = ['age', 'fnlwgt', 'education-num', 'capital-gain',
                 'capital-loss', 'hours-per-week']

print(f"\n3. FEATURES")
print(f"   - Numerical features: {len(numerical_cols)}")
for col in numerical_cols:
    print(f"     * {col}")
print(f"   - Categorical features: {len(categorical_cols)}")
for col in categorical_cols:
    print(f"     * {col} ({df_clean[col].nunique()} unique values)")

# Encode categorical variables
print(f"\n4. PREPROCESSING")
print(f"   - Encoding categorical variables...")
df_encoded = df_clean.copy()
label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df_encoded[col])
    label_encoders[col] = le

# Encode target
le_target = LabelEncoder()
df_encoded['income'] = le_target.fit_transform(df_encoded['income'])

# Split features and target
X = df_encoded.drop('income', axis=1)
y = df_encoded['income']

# Karena dataset besar, gunakan subset untuk KNN (KNN lambat untuk data besar)
# Ambil 10% data untuk efisiensi komputasi
print(f"   - Using subset of data for KNN (10% sample for efficiency)")
X_subset, _, y_subset, _ = train_test_split(
    X, y, train_size=0.1, random_state=42, stratify=y
)
print(f"   - Subset size: {len(X_subset)} samples")

# Split train-test (80-20)
X_train, X_test, y_train, y_test = train_test_split(
    X_subset, y_subset, test_size=0.2, random_state=42, stratify=y_subset
)

print(f"\n5. TRAIN-TEST SPLIT")
print(f"   - Training set: {len(X_train)} samples")
print(f"   - Test set: {len(X_test)} samples")

# Standardization
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"   - Standardization: Applied (mean=0, std=1)")

# Train KNN dengan k=5 (baseline)
k = 5
print(f"\n6. MODEL TRAINING")
print(f"   - Algorithm: K-Nearest Neighbors")
print(f"   - K value: {k}")
print(f"   - Distance metric: Euclidean")
print(f"   - Training... (may take a while)")

knn = KNeighborsClassifier(n_neighbors=k, metric='euclidean', n_jobs=-1)
knn.fit(X_train_scaled, y_train)

# Prediction
y_pred = knn.predict(X_test_scaled)

# Evaluation
print(f"\n7. EVALUATION RESULTS")
print(f"\n   Test Set Performance:")
accuracy = accuracy_score(y_test, y_pred)
print(f"   - Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

print(f"\n   Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(f"   {cm}")

print(f"\n   Classification Report:")
print(classification_report(y_test, y_pred, target_names=['<=50K', '>50K']))

# Cross-validation (menggunakan cv=3 untuk efisiensi)
print(f"\n   3-Fold Cross-Validation (on training set):")
print(f"   - Running... (may take a while)")
cv_scores = cross_val_score(knn, X_train_scaled, y_train, cv=3,
                           scoring='accuracy', n_jobs=-1)
print(f"   - Scores: {cv_scores}")
print(f"   - Mean CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# Eksperimen dengan berbagai nilai K
print(f"\n8. K-VALUE COMPARISON")
k_values = [3, 5, 7, 9, 11]
k_scores = []

for k_val in k_values:
    print(f"   - Testing k={k_val}...")
    knn_temp = KNeighborsClassifier(n_neighbors=k_val, n_jobs=-1)
    knn_temp.fit(X_train_scaled, y_train)
    score = knn_temp.score(X_test_scaled, y_test)
    k_scores.append(score)
    print(f"     k={k_val:2d} -> Accuracy: {score:.4f}")

best_k = k_values[np.argmax(k_scores)]
best_score = max(k_scores)
print(f"\n   Best K: {best_k} with accuracy {best_score:.4f}")

print("\n" + "=" * 60)
print("BASELINE KNN TRAINING COMPLETED")
print(f"Note: This analysis used 10% of the full dataset for efficiency.")
print(f"KNN can be slow on large datasets. Consider using other algorithms")
print(f"for full dataset (e.g., Decision Trees, Random Forest, etc.)")
print("=" * 60)
