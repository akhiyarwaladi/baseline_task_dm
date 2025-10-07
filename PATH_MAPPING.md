# 🗺️ Complete Path Mapping Reference

Last Updated: 2025-10-07

## 📊 Unified Numbering System

All files use consistent **01-04** numbering:

| No | Topic | Script | App | Dataset | Model |
|----|-------|--------|-----|---------|-------|
| **01** | Stunting Detection | ✓ | ✓ | ✓ | ✓ |
| **02** | SMS Spam Classification | ✓ | ✓ | ✓ | ✓ |
| **03** | Emotion Classification | ✓ | ✓ | ✓ | ✓ |
| **04** | Customer Churn Prediction | ✓ | ✓ | ✓ | ✓ |

---

## 📁 File Paths by Number

### 01 - Stunting Detection

```
Script:   scripts/01_baseline_stunting.py
App:      apps/01_app_stunting.py  
Dataset:  dataset/01_stunting_balita.csv
Model:    models/01_model_stunting.pkl

Dataset Size:   120,999 samples (3.2MB)
Algorithm:      KNN (K=5, Euclidean)
Accuracy:       99.67%
Classes:        4 (Normal, Stunted, Severely Stunted, Tinggi)
```

### 02 - SMS Spam Classification

```
Script:   scripts/02_baseline_sms_spam.py
App:      apps/02_app_sms_spam.py
Dataset:  dataset/02_sms_spam.csv
Model:    models/02_model_sms_spam.pkl

Dataset Size:   1,143 SMS (127KB)
Algorithm:      KNN (K=5, Cosine + TF-IDF 500)
Accuracy:       89.08%
Classes:        3 (Ham, Promosi, Spam)
```

### 03 - Emotion Classification

```
Script:   scripts/03_baseline_emotion.py
App:      apps/03_app_emotion.py
Dataset:  dataset/03_tokopedia_emotion.csv
Model:    models/03_model_emotion.pkl

Dataset Size:   5,400 reviews (1.3MB)
Algorithm:      KNN (K=7, Cosine + TF-IDF 1000)
Accuracy:       53.15%
Classes:        5 (Happy, Sadness, Anger, Fear, Love)
```

### 04 - Customer Churn Prediction

```
Script:   scripts/04_baseline_churn.py
App:      apps/04_app_churn.py
Dataset:  dataset/04_ecommerce_churn.csv
Model:    models/04_model_churn.pkl

Dataset Size:   5,630 customers (561KB)
Algorithm:      KNN (K=9, Euclidean + StandardScaler)
Accuracy:       93.52%, ROC-AUC: 0.9705
Classes:        2 (Churn, No Churn)
```

---

## 🔄 How Files Reference Each Other

### Scripts → Datasets

All scripts in `scripts/` folder reference datasets with **relative path**:

```python
# scripts/01_baseline_stunting.py
df = pd.read_csv('../dataset/01_stunting_balita.csv')

# scripts/02_baseline_sms_spam.py
df = pd.read_csv('../dataset/02_sms_spam.csv')

# scripts/03_baseline_emotion.py
df = pd.read_csv('../dataset/03_tokopedia_emotion.csv')

# scripts/04_baseline_churn.py
df = pd.read_csv('../dataset/04_ecommerce_churn.csv')
```

### Apps → Datasets

All apps in `apps/` folder reference datasets with **relative path**:

```python
# apps/01_app_stunting.py
df = pd.read_csv('dataset/01_stunting_balita.csv')

# apps/02_app_sms_spam.py
df = pd.read_csv('dataset/02_sms_spam.csv')

# apps/03_app_emotion.py
df = pd.read_csv('dataset/03_tokopedia_emotion.csv')

# apps/04_app_churn.py
df = pd.read_csv('dataset/04_ecommerce_churn.csv')
```

### Apps → Models

All apps auto-save/load models to `models/` folder:

```python
# apps/01_app_stunting.py
model_path = 'models/01_model_stunting.pkl'

# apps/02_app_sms_spam.py
model_path = 'models/02_model_sms_spam.pkl'

# apps/03_app_emotion.py
model_path = 'models/03_model_emotion.pkl'

# apps/04_app_churn.py
model_path = 'models/04_model_churn.pkl'
```

---

## 🚀 Usage Workflows

### Workflow 1: Run Training Script First

```bash
# 1. Train model with script (evaluation only, no model saved)
python scripts/01_baseline_stunting.py

# 2. Run app (will auto-train and save model if not exists)
streamlit run apps/01_app_stunting.py

# Model saved to: models/01_model_stunting.pkl
```

### Workflow 2: Run App Directly (Recommended)

```bash
# App will:
# 1. Check if models/01_model_stunting.pkl exists
# 2. If NOT: auto-train from dataset/01_stunting_balita.csv
# 3. Save to models/01_model_stunting.pkl
# 4. Use model for predictions

streamlit run apps/01_app_stunting.py
```

### Workflow 3: Delete Model to Retrain

```bash
# Delete old model
rm models/01_model_stunting.pkl

# Restart app (will retrain automatically)
streamlit run apps/01_app_stunting.py
```

---

## 📊 Directory Structure

```
baseline_task/
├── scripts/
│   ├── 01_baseline_stunting.py     # Uses ../dataset/01_*.csv
│   ├── 02_baseline_sms_spam.py     # Uses ../dataset/02_*.csv
│   ├── 03_baseline_emotion.py      # Uses ../dataset/03_*.csv
│   └── 04_baseline_churn.py        # Uses ../dataset/04_*.csv
│
├── apps/
│   ├── 01_app_stunting.py          # Uses dataset/01_*.csv, saves to models/01_*.pkl
│   ├── 02_app_sms_spam.py          # Uses dataset/02_*.csv, saves to models/02_*.pkl
│   ├── 03_app_emotion.py           # Uses dataset/03_*.csv, saves to models/03_*.pkl
│   └── 04_app_churn.py             # Uses dataset/04_*.csv, saves to models/04_*.pkl
│
├── dataset/
│   ├── 01_stunting_balita.csv
│   ├── 02_sms_spam.csv
│   ├── 03_tokopedia_emotion.csv
│   ├── 04_ecommerce_churn.csv
│   └── 05_jakarta_air_quality.csv  # For practice exercise
│
├── models/                          # Auto-created by apps
│   ├── 01_model_stunting.pkl       # Created by 01_app_stunting.py
│   ├── 02_model_sms_spam.pkl       # Created by 02_app_sms_spam.py
│   ├── 03_model_emotion.pkl        # Created by 03_app_emotion.py
│   └── 04_model_churn.pkl          # Created by 04_app_churn.py
│
├── utils.py                         # Shared utilities for scripts
└── apps/app_utils.py                # Shared utilities for Streamlit apps
```

---

## ✅ Consistency Checklist

- [x] All numbered 01-04 consistently
- [x] Scripts use `../dataset/0X_*.csv`
- [x] Apps use `dataset/0X_*.csv`
- [x] Models saved to `models/0X_model_*.pkl`
- [x] Easy to identify which files belong together
- [x] No hardcoded absolute paths
- [x] Models auto-created on first app run
- [x] Clean separation of concerns

---

## 🔍 Troubleshooting

### Model not found error

```
FileNotFoundError: [Errno 2] No such file or directory: 'models/01_model_stunting.pkl'
```

**Solution:** Normal behavior! App will auto-train on first run. Wait ~30 seconds.

### Dataset not found error

```
FileNotFoundError: [Errno 2] No such file or directory: 'dataset/01_stunting_balita.csv'
```

**Solution:** Make sure you're running from the **root project directory**:

```bash
cd /path/to/baseline_task
streamlit run apps/01_app_stunting.py
```

### Script dataset not found

```bash
# WRONG (from root)
python scripts/01_baseline_stunting.py
# Error: FileNotFoundError: '../dataset/01_stunting_balita.csv'

# CORRECT (from scripts directory)
cd scripts
python 01_baseline_stunting.py
```

---

**Dibuat dengan ❤️ untuk consistency and clarity**
