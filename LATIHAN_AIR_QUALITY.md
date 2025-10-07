# 🌫️ LATIHAN: Jakarta Air Quality Prediction

## 📋 Deskripsi Dataset

**Dataset:** Air Quality Index di Jakarta (2010-2021)
**File:** `dataset/05_jakarta_air_quality.csv`
**Samples:** 5,538 pengukuran harian
**Source:** Kaggle - Air Quality Index in Jakarta

## 📊 Informasi Dataset

### Kolom Dataset:
1. **tanggal** - Tanggal pengukuran (2010-2021)
2. **stasiun** - Stasiun pengukuran (DKI1-DKI5)
3. **pm25** - Particulate Matter 2.5 µm (µg/m³)
4. **pm10** - Particulate Matter 10 µm (µg/m³)
5. **so2** - Sulfur Dioxide (µg/m³)
6. **co** - Carbon Monoxide (µg/m³)
7. **o3** - Ozone (µg/m³)
8. **no2** - Nitrogen Dioxide (µg/m³)
9. **max** - Nilai ISPU tertinggi
10. **critical** - Polutan kritis (penyumbang tertinggi)
11. **categori** - Kategori kualitas udara

### Target (Kategori Kualitas Udara):
- **BAIK** - Kualitas udara baik, tidak ada dampak kesehatan
- **SEDANG** - Kualitas udara sedang, dapat berdampak pada kelompok sensitif
- **TIDAK SEHAT** - Mulai berdampak pada kesehatan
- **SANGAT TIDAK SEHAT** - Berbahaya bagi kesehatan
- **BERBAHAYA** - Sangat berbahaya

## 🎯 Tugas Latihan

Buat baseline model KNN untuk memprediksi **kategori kualitas udara** berdasarkan data polutan.

### Task 1: Binary Classification (MUDAH)
Prediksi: **Sehat (BAIK + SEDANG)** vs **Tidak Sehat (lainnya)**

**Steps:**
1. Load dataset `05_jakarta_air_quality.csv`
2. Handle missing values (pm25 banyak NaN)
3. Create binary target: `is_unhealthy`
4. Feature selection: pm10, so2, co, o3, no2
5. Split train/test (80/20)
6. Train KNN model (pilih K yang sesuai)
7. Evaluate: accuracy, confusion matrix, classification report

### Task 2: Multi-class Classification (SEDANG)
Prediksi: **5 kategori** (BAIK, SEDANG, TIDAK SEHAT, dll)

**Additional Steps:**
- Handle imbalanced classes (BAIK >> BERBAHAYA)
- Feature engineering: month, year dari tanggal
- Try different K values (3, 5, 7, 9)
- Cross-validation

### Task 3: Streamlit Dashboard (LANJUT)
Buat dashboard untuk prediksi real-time kualitas udara

**Features:**
- Input: pm10, so2, co, o3, no2
- Output: Kategori + Rekomendasi kesehatan
- Visualization: Trend polutan
- Warning system untuk kategori tidak sehat

## 📝 Panduan Pengerjaan

### 1. Explorasi Data
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv('dataset/05_jakarta_air_quality.csv')

# TODO: Eksplorasi dataset
# - Cek missing values
# - Distribusi kategori
# - Korelasi antar features
# - Trend PM2.5/PM10 over time
```

### 2. Preprocessing
```python
# TODO: Data cleaning
# - Handle missing values (strategi: drop/impute?)
# - Convert tanggal to datetime
# - Extract features: year, month, day_of_week
# - Encode categorical: stasiun
# - Create target variable
```

### 3. Model Training
```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# TODO: Split data

# TODO: Preprocessing (scaling important!)

# TODO: Train KNN
# - Pilih K yang optimal
# - Distance metric: euclidean/manhattan?
# - Weights: uniform/distance?
```

### 4. Evaluation
```python
from sklearn.metrics import classification_report, confusion_matrix

# TODO: Evaluate model
# - Accuracy
# - Precision, Recall, F1 per class
# - Confusion matrix
# - Cross-validation score
```

## 💡 Tips & Hints

### Data Preprocessing:
- ⚠️ PM2.5 banyak missing → drop column atau impute?
- 📅 Tanggal → extract temporal features (month bisa penting!)
- 📍 Stasiun → encode as categorical
- ⚖️ Imbalanced classes → consider stratified split

### Feature Engineering:
- Musim kemarau vs hujan (bulan 4-9 vs 10-3)?
- Ratio CO/O3?
- Moving average PM10?
- Weekend vs weekday?

### Model Selection:
- K terlalu kecil → overfitting
- K terlalu besar → underfitting
- Coba K = [3, 5, 7, 9, 11] dan bandingkan
- Scaling WAJIB untuk KNN!

### Evaluation:
- Fokus pada kategori "TIDAK SEHAT" keatas (bahaya kesehatan!)
- False Negative (miss kategori berbahaya) lebih buruk dari False Positive
- Consider precision untuk kategori berbahaya

## 🎯 Expected Results

### Binary Classification:
- **Target Accuracy**: >85%
- **ROC-AUC**: >0.90

### Multi-class Classification:
- **Target Accuracy**: >70%
- **F1-Score (weighted)**: >0.70

## 📚 Resources

### Referensi:
- ISPU (Indeks Standar Pencemar Udara)
- WHO Air Quality Guidelines
- scikit-learn KNN Documentation

### Batas ISPU:
- BAIK: 0-50
- SEDANG: 51-100
- TIDAK SEHAT: 101-200
- SANGAT TIDAK SEHAT: 201-300
- BERBAHAYA: 300+

## ✅ Deliverables

1. **Script Python** (`08_baseline_air_quality.py`)
   - Data loading & preprocessing
   - Model training
   - Evaluation & visualization

2. **Streamlit Dashboard** (`05_app_air_quality.py`) [OPTIONAL]
   - Input form untuk polutan
   - Prediksi kategori
   - Rekomendasi kesehatan
   - Trend visualization

3. **Report Singkat** (Markdown/PDF)
   - Methodology
   - Results & discussion
   - Insights & recommendations

## 🚀 Bonus Challenges

1. **Time Series Split**: Train on 2010-2019, test on 2020-2021
2. **Location Analysis**: Compare accuracy per stasiun
3. **Feature Importance**: Which pollutant matters most?
4. **Ensemble**: Combine KNN with other models
5. **API Development**: Deploy model sebagai REST API

---

## ⚠️ PERHATIAN

**JANGAN LANGSUNG LIHAT JAWABAN!**

Coba kerjakan sendiri dulu. Ini kesempatan belajar:
- Problem solving
- Data preprocessing
- Feature engineering
- Model tuning
- Evaluation interpretation

Stuck? Check hints di section Tips & Hints!

---

**Selamat Mengerjakan! 🎓**

*Catatan: Dataset ini real data Jakarta. Insights yang kamu dapat bisa berguna untuk kesehatan masyarakat!*
