# Baseline KNN - Tutorial Data Mining

Project ini berisi implementasi baseline K-Nearest Neighbors (KNN) untuk 3 dataset dari UCI Machine Learning Repository.

## Dataset

1. **Heart Disease Dataset** - Prediksi penyakit jantung (303 samples, 14 features)
2. **Wine Recognition Dataset** - Klasifikasi jenis wine (178 samples, 13 features)
3. **Adult Income Dataset** - Prediksi income >50K (48842 samples, 14 features)

## Setup

### 1. Aktivasi Virtual Environment

```bash
# Aktivasi venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Jalankan Script

**Heart Disease:**
```bash
python baseline_knn_heart_disease.py
```

**Wine Recognition:**
```bash
python baseline_knn_wine.py
```

**Adult Income:**
```bash
python baseline_knn_adult.py
```

## Output

Setiap script akan menampilkan:
- ✅ Data loading dan preprocessing
- ✅ Distribusi target/class
- ✅ Train-test split (80-20)
- ✅ Model training dengan K=5
- ✅ Evaluation metrics (accuracy, confusion matrix, classification report)
- ✅ 5-fold cross-validation
- ✅ Perbandingan berbagai nilai K

## Catatan

- **Heart Disease & Wine**: Menggunakan full dataset
- **Adult**: Menggunakan 10% subset untuk efisiensi (KNN lambat untuk dataset besar)
- Semua feature dinormalisasi dengan StandardScaler (penting untuk KNN)
- Default K=5 untuk baseline

## Dependencies

- pandas 2.2.0
- numpy 1.26.3
- scikit-learn 1.4.0

---
Tutorial Data Mining - Baseline KNN
# baseline_task_dm
