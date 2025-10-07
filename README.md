# 📊 Baseline KNN Models - Data Mining Project

Koleksi lengkap baseline model KNN untuk berbagai dataset dengan Streamlit dashboard interaktif.

## 📁 Struktur Folder

```
baseline_task/
├── apps/                           # Streamlit dashboards
│   ├── 01_app_stunting.py         # Dashboard stunting detection
│   ├── 02_app_sms_spam.py         # Dashboard SMS spam
│   ├── 03_app_emotion.py          # Dashboard emotion detection
│   └── 04_app_churn.py            # Dashboard customer churn
│
├── scripts/                        # Baseline training scripts
│   ├── 01_baseline_adult.py       # Adult income prediction
│   ├── 02_baseline_heart_disease.py
│   ├── 03_baseline_wine.py
│   ├── 04_baseline_stunting.py    # Stunting detection
│   ├── 05_baseline_sms_spam.py    # SMS spam classification
│   ├── 06_baseline_emotion.py     # Emotion classification
│   └── 07_baseline_churn.py       # Customer churn prediction
│
├── dataset/                        # Clean datasets
│   ├── 01_stunting_balita.csv     # 120K balita (3.2MB)
│   ├── 02_sms_spam.csv            # 1.1K SMS (127KB)
│   ├── 03_tokopedia_emotion.csv   # 5.4K reviews (1.3MB)
│   └── 04_ecommerce_churn.csv     # 5.6K customers (561KB)
│
├── models/                         # Saved trained models (.pkl)
├── archive/                        # Old datasets & files
└── README.md
```

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Baseline Training
```bash
# Training & evaluation
python scripts/04_baseline_stunting.py
python scripts/05_baseline_sms_spam.py
python scripts/06_baseline_emotion.py
python scripts/07_baseline_churn.py
```

### 3. Run Streamlit Dashboard
```bash
# Interactive prediction interface
streamlit run apps/01_app_stunting.py
streamlit run apps/02_app_sms_spam.py
streamlit run apps/03_app_emotion.py
streamlit run apps/04_app_churn.py
```

Dashboard: http://localhost:8501

---

## 📊 Dataset & Model Performance

### 1. 👶 Stunting Detection
- **Dataset**: 120,999 balita Indonesia
- **Task**: 4-class classification
- **Accuracy**: 99.67% ⭐
- **Classes**: Normal, Severely Stunted, Stunted, Tinggi

### 2. 📱 SMS Spam Detection  
- **Dataset**: 1,143 SMS berbahasa Indonesia
- **Task**: 3-class classification
- **Accuracy**: 89.08%
- **Classes**: Ham (Normal), Promosi, Spam

### 3. 😊 Tokopedia Review Emotion
- **Dataset**: 5,400 review produk (PRDECT-ID)
- **Task**: 5-class emotion classification
- **Accuracy**: 53.15%
- **Classes**: Happy, Sadness, Anger, Fear, Love

### 4. 📊 E-Commerce Customer Churn
- **Dataset**: 5,630 customers, 20 features
- **Task**: Binary classification
- **Accuracy**: 93.52%, ROC-AUC: 0.9705 ⭐
- **Classes**: Churn, No Churn

---

## 🤖 Model Specifications

| Dataset | Algorithm | K | Distance | Preprocessing |
|---------|-----------|---|----------|---------------|
| Stunting | KNN | 5 | Euclidean | One-hot encoding |
| SMS Spam | KNN | 5 | Cosine | TF-IDF (500 features) |
| Emotion | KNN | 7 | Cosine | TF-IDF (1000 features) |
| Churn | KNN | 9 | Euclidean | StandardScaler + Label Encoding |

---

## 💡 Use Cases

### 👶 Stunting Detection
- Screening balita di posyandu
- Early warning system
- Monitoring tumbuh kembang anak

### 📱 SMS Spam Detection
- Filter SMS spam otomatis
- Protect users dari phishing
- Prioritas inbox management

### 😊 Emotion Detection
- Seller monitoring customer sentiment
- Prioritas response untuk angry customers
- Product improvement insights
- Review analysis automation

### 📊 Customer Churn
- Identify high-risk customers
- Targeted retention campaign
- Reduce customer acquisition cost
- Proactive customer engagement

---

## 🎯 Dashboard Features

Setiap dashboard memiliki 3 menu:

### 🏠 Home
- Overview dataset & model performance
- Key metrics
- Feature descriptions

### 🔍 Prediksi
- Input form untuk data baru
- Real-time prediction
- Confidence score & probability distribution
- Actionable recommendations

### 📈 Model Info
- Model specifications
- Performance metrics
- Data distribution
- Usage instructions

---

## 🛠️ Tech Stack

- **Python 3.x**
- **scikit-learn** - Machine Learning
- **pandas & numpy** - Data manipulation
- **streamlit** - Interactive dashboards
- **TF-IDF** - Text vectorization
- **pickle** - Model persistence

---

## 📝 Notes

- Models auto-train on first dashboard run
- Saved to `models/*.pkl` for fast loading
- Dataset harus ada di folder `dataset/`
- Multiple dashboards dapat run bersamaan di port berbeda

---

## 📚 Practice Exercises

Soal latihan untuk belajar mandiri (tanpa solusi):

### 🌫️ Air Quality Prediction
- **File**: `LATIHAN_AIR_QUALITY.md`
- **Dataset**: Jakarta Air Quality (5,538 samples)
- **Task**: KNN untuk prediksi kategori kualitas udara
- **Level**: Binary (Sehat vs Tidak Sehat) + Multi-class (5 kategori)

### 🌾 Crop Recommendation
- **File**: `LATIHAN_CROP_RECOMMENDATION.md`
- **Dataset**: Crop Recommendation (2,200 samples)
- **Task**: Decision Tree untuk rekomendasi tanaman pertanian
- **Level**: Binary, 4-class, dan 22-class classification
- **Focus**: Interpretability & feature importance

---

## 🤝 Contributing

Contributions welcome! Feel free to:
- Add new datasets
- Improve model performance
- Enhance dashboard UI/UX
- Fix bugs

---

## 📄 License

Educational Purpose - Data Mining Project

---

**Dibuat dengan ❤️ menggunakan Python, scikit-learn, dan Streamlit**
