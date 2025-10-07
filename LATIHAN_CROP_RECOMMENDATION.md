# 🌾 LATIHAN: Crop Recommendation System

## 📋 Deskripsi Dataset

**Dataset:** Crop Recommendation Dataset
**File:** `Crop_recommendation.csv`
**Samples:** 2,200 sampel pertanian
**Source:** Kaggle - Crop Recommendation Dataset
**Link:** https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset

## 📊 Informasi Dataset

### Fitur Dataset (8 fitur):

**Soil Nutrients (Nutrisi Tanah):**
1. **N** - Nitrogen content ratio (kg/hektar)
2. **P** - Phosphorus content ratio (kg/hektar)
3. **K** - Potassium content ratio (kg/hektar)
4. **ph** - pH tanah (range: 5.5-8.5)

**Climate Variables (Variabel Iklim):**
5. **temperature** - Suhu (°C, range: 10-45°C)
6. **humidity** - Kelembaban relatif (%, range: 10-100%)
7. **rainfall** - Curah hujan (mm, range: 20-300mm)

### Target Variable:

**label** - Jenis tanaman yang direkomendasikan (22 kelas):

**Cereals (Sereal):**
- Rice (Padi)
- Maize (Jagung)

**Pulses (Kacang-kacangan):**
- Chickpea, Kidney Beans, Pigeon Peas, Moth Beans, Mung Bean, Black Gram, Lentil

**Fruits (Buah-buahan):**
- Pomegranate, Banana, Mango, Grapes, Watermelon, Muskmelon, Apple, Orange, Papaya, Coconut

**Commercial Crops (Tanaman Komersial):**
- Cotton, Jute, Coffee

## 🎯 Tugas Latihan

Buat baseline model Decision Tree untuk merekomendasikan **jenis tanaman** berdasarkan kondisi tanah dan iklim.

### Task 1: Multi-class Classification (SEDANG)
Prediksi: **22 jenis tanaman** (full classification)

**Steps:**
1. Download dataset dari Kaggle
2. Load dataset `Crop_recommendation.csv`
3. Explorasi data (distribusi kelas, korelasi fitur)
4. Feature selection: semua 7 fitur (N, P, K, ph, temperature, humidity, rainfall)
5. Split train/test (80/20)
6. Train Decision Tree model
7. Visualisasi tree (text format dan PNG)
8. Evaluate: accuracy, confusion matrix, classification report

### Task 2: Binary Classification (MUDAH)
Prediksi: **Cereal vs Non-cereal**

**Additional Steps:**
- Create binary target: `is_cereal` (Rice/Maize vs lainnya)
- Interpretasi decision rules untuk perbedaan tanaman sereal
- Feature importance analysis

### Task 3: 4-Class Classification (LANJUT)
Prediksi: **4 kategori tanaman**

**Categories:**
- **Cereals**: Rice, Maize
- **Pulses**: Chickpea, Kidney Beans, dll
- **Fruits**: Banana, Mango, dll
- **Commercial**: Cotton, Jute, Coffee

**Features:**
- Group crops into 4 categories
- Compare decision rules antar kategori
- Visualisasi decision tree dengan max_depth=4

## 📝 Panduan Pengerjaan

### 1. Download & Load Data

```python
# Di Google Colab
!pip install kaggle -q
!mkdir -p ~/.kaggle
# Upload kaggle.json ke ~/.kaggle/

!kaggle datasets download -d atharvaingle/crop-recommendation-dataset
!unzip crop-recommendation-dataset.zip

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Load data
df = pd.read_csv('Crop_recommendation.csv')

# TODO: Eksplorasi dataset
# - Shape dan info
# - Distribusi kelas
# - Statistik deskriptif
# - Korelasi antar fitur
```

### 2. Explorasi Data

```python
# TODO: Dataset overview
print(f"Shape: {df.shape}")
print(f"\nColumns: {df.columns.tolist()}")
print(f"\nJumlah crops: {df['label'].nunique()}")

# TODO: Cek distribusi kelas
print("\nDistribusi crops:")
print(df['label'].value_counts())

# TODO: Cek NPK ratios untuk tanaman tertentu
rice_df = df[df['label'] == 'rice']
print(f"\nRice NPK ranges:")
print(f"N: {rice_df['N'].min():.0f}-{rice_df['N'].max():.0f}")
print(f"P: {rice_df['P'].min():.0f}-{rice_df['P'].max():.0f}")
print(f"K: {rice_df['K'].min():.0f}-{rice_df['K'].max():.0f}")

# TODO: Visualisasi
# - Distribusi NPK
# - Temperature vs Rainfall
# - Feature correlation heatmap
```

### 3. Preprocessing

```python
# TODO: Prepare features and target
X = df.drop('label', axis=1)
y = df['label']

# TODO: Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")
```

### 4. Model Training

```python
# TODO: Train Decision Tree
dt = DecisionTreeClassifier(
    max_depth=5,        # Limit depth for interpretability
    min_samples_split=20,
    min_samples_leaf=10,
    random_state=42
)

dt.fit(X_train, y_train)

# TODO: Evaluate
y_pred = dt.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\nAccuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
```

### 5. Tree Visualization

```python
# TODO: Text visualization
tree_rules = export_text(dt, feature_names=list(X.columns))
print(tree_rules)

# TODO: PNG visualization
plt.figure(figsize=(20, 12))
plot_tree(dt,
          feature_names=X.columns,
          class_names=sorted(y.unique()),
          filled=True,
          rounded=True,
          fontsize=8)
plt.title('Decision Tree - Crop Recommendation')
plt.tight_layout()
plt.savefig('crop_decision_tree.png', dpi=300)
plt.show()
```

### 6. Feature Importance

```python
# TODO: Analyze feature importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': dt.feature_importances_
}).sort_values('importance', ascending=False)

print("\nFeature Importance:")
print(feature_importance)

# TODO: Visualize
plt.figure(figsize=(10, 6))
plt.barh(feature_importance['feature'], feature_importance['importance'])
plt.xlabel('Importance')
plt.title('Feature Importance for Crop Recommendation')
plt.tight_layout()
plt.show()
```

## 💡 Tips & Hints

### Data Understanding:
- 📊 Dataset **balanced** - setiap crop punya ~100 samples
- 🌱 NPK ratio berbeda signifikan antar crops
- 🌡️ Temperature dan rainfall adalah key factors untuk tropical crops
- 💧 Rice butuh high rainfall, cotton butuh low humidity

### Decision Tree Insights:
- Fitur NPK biasanya paling important (nutrisi tanah critical)
- pH bisa jadi root split (asam vs alkali loving plants)
- Temperature splits tropical vs temperate crops
- Rainfall splits rice/jute (high water) vs cotton (low water)

### Model Tuning:
- `max_depth=3-5` untuk interpretability
- `max_depth=8-10` untuk maximum accuracy
- `min_samples_leaf=10-20` untuk avoid overfitting
- Gunakan `class_weight='balanced'` jika data imbalanced

### Real-world Rules:
- Rice: N>60, K>35, rainfall>150mm
- Cotton: humidity<70%, rainfall<100mm, temperature>20°C
- Coffee: ph=6-7, temperature=20-25°C
- Maize: N>70, temperature>18°C

## 🎯 Expected Results

### Full Classification (22 classes):
- **Target Accuracy**: >90%
- **F1-Score (weighted)**: >0.90

### Binary Classification (Cereal vs Non-cereal):
- **Target Accuracy**: >95%
- **ROC-AUC**: >0.98

### 4-Class Classification:
- **Target Accuracy**: >95%
- **F1-Score (weighted)**: >0.95

## 🌾 Relevansi dengan Indonesia

### Mengapa Dataset Ini Penting:

1. **Food Security**
   - Indonesia produsen beras terbesar ke-3 dunia
   - Optimasi produksi padi crucial untuk ketahanan pangan
   - Rekomendasi crop dapat maksimalkan yield

2. **Precision Agriculture**
   - Petani dapat tes tanah → dapat rekomendasi optimal crop
   - Hindari crop failure dengan match soil/climate
   - Sustainable farming practices

3. **Climate Adaptation**
   - Perubahan iklim → butuh adapt crop choices
   - Decision tree dapat guide planting decisions
   - Resource optimization (pupuk, air)

4. **Economic Impact**
   - Pertanian = 13% GDP Indonesia
   - Right crop = higher income untuk petani
   - Reduce waste dari failed crops

### Indonesian Crops in Dataset:
- ✅ Rice (Padi) - Staple food
- ✅ Maize (Jagung) - 2nd staple
- ✅ Coconut (Kelapa) - Major export
- ✅ Banana (Pisang) - Popular fruit
- ✅ Coffee (Kopi) - Export commodity

## 📚 Resources

### Referensi Agronomis:
- NPK Ratios untuk berbagai tanaman
- Soil pH requirements
- Climate zones Indonesia
- FAO crop requirements

### Technical References:
- Decision Tree for agriculture
- Feature importance interpretation
- Precision agriculture papers

## ✅ Deliverables

1. **Script Python** (`08_baseline_crop_recommendation.py`)
   - Data loading & exploration
   - Decision Tree training
   - Tree visualization (text + PNG)
   - Evaluation & interpretation

2. **Notebook** (`LATIHAN_Crop_Recommendation.ipynb`) [OPTIONAL]
   - Step-by-step dengan visualisasi
   - Exploratory data analysis
   - Model interpretation
   - Real-world insights

3. **Report Singkat** (Markdown/PDF)
   - Dataset insights
   - Decision rules interpretation
   - Feature importance analysis
   - Recommendations untuk petani Indonesia

## 🚀 Bonus Challenges

1. **Crop Grouping**: Buat classification untuk crop families (cereals, pulses, fruits, commercial)
2. **Regional Analysis**: Filter untuk crops yang cocok di Indonesia (tropical)
3. **Season Recommendation**: Tambah fitur season/month untuk timing
4. **Yield Prediction**: Extend model untuk predict expected yield
5. **Interactive Tool**: Buat Streamlit app untuk farmers

## ⚠️ PERHATIAN

**JANGAN LANGSUNG LIHAT JAWABAN!**

Ini kesempatan belajar:
- Decision tree untuk real agricultural problem
- Feature importance interpretation
- Translating model ke actionable insights
- Understanding agronomic principles

Stuck? Check tips di section Tips & Hints!

---

## 💡 Decision Tree Learning Goals

### Konsep yang Dipelajari:

1. **Tree Structure**
   - Root node selection (most informative feature)
   - Internal nodes (decision splits)
   - Leaf nodes (final predictions)
   - Depth vs accuracy tradeoff

2. **Split Criteria**
   - Gini impurity
   - Information gain
   - Feature thresholds

3. **Interpretability**
   - Decision paths as agronomic rules
   - Feature importance rankings
   - Visual tree representation

4. **Practical Application**
   - Converting model ke farming advice
   - Validating rules dengan domain knowledge
   - Communicating insights ke non-technical users (farmers)

---

**Selamat Mengerjakan! 🎓🌾**

*Catatan: Dataset ini real agricultural data. Insights yang kamu dapat bisa membantu petani Indonesia membuat keputusan tanam yang lebih baik!*
