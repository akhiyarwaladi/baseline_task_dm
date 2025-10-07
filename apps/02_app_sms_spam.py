"""
Streamlit Dashboard - SMS Spam Detection
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
import pickle
import os
from app_utils import create_sidebar_menu, display_metrics, display_footer, create_prediction_button

# Page config
st.set_page_config(page_title="SMS Spam Detection", page_icon="📱", layout="wide")

st.title("📱 SMS Spam Detection Indonesia")
st.markdown("**Model KNN untuk klasifikasi SMS spam/promosi/normal**")

# Sidebar menu
menu = create_sidebar_menu()

# Load or train model
@st.cache_resource
def load_model():
    model_path = 'models/02_model_sms_spam.pkl'

    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            return pickle.load(f)

    df = pd.read_csv('dataset/02_sms_spam.csv')
    X, y = df['Teks'], df['label']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    vectorizer = TfidfVectorizer(max_features=500, ngram_range=(1, 2))
    X_train_tfidf = vectorizer.fit_transform(X_train)

    knn = KNeighborsClassifier(n_neighbors=5, metric='cosine')
    knn.fit(X_train_tfidf, y_train)

    os.makedirs('models', exist_ok=True)
    with open(model_path, 'wb') as f:
        pickle.dump({'model': knn, 'vectorizer': vectorizer}, f)

    return {'model': knn, 'vectorizer': vectorizer}

model_data = load_model()
model, vectorizer = model_data['model'], model_data['vectorizer']

# Label info
LABEL_INFO = {
    0: {'name': 'HAM (Normal)', 'emoji': '✅', 'desc': 'SMS Personal'},
    1: {'name': 'PROMOSI (Iklan)', 'emoji': '📢', 'desc': 'SMS Marketing'},
    2: {'name': 'SPAM (Phishing)', 'emoji': '🚨', 'desc': 'SMS Penipuan'}
}

# ============================================================================
# HOME
# ============================================================================
if menu == "🏠 Home":
    st.header("Selamat Datang! 👋")

    display_metrics([
        ("Total SMS", "1,143"),
        ("Akurasi Model", "89.08%"),
        ("Algoritma", "KNN (K=5)")
    ])

    st.markdown("---")

    st.subheader("📋 Tentang Dataset")
    st.write("""
    Dataset SMS berbahasa Indonesia dengan 3 kategori:

    - ✅ **Ham (Normal)** - Personal (49.8%)
    - 📢 **Promosi (Iklan)** - Marketing (29.3%)
    - 🚨 **Spam (Phishing)** - Penipuan (20.9%)
    """)

    st.subheader("🔍 Contoh SMS")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.success("**✅ HAM**")
        st.write("_'Oke, nanti aku jemput jam 8 ya'_")

    with col2:
        st.info("**📢 PROMOSI**")
        st.write("_'Paket 10GB hanya 50rb. Hub *123#'_")

    with col3:
        st.error("**🚨 SPAM**")
        st.write("_'GRATIS pulsa 100rb! Klik link'_")

# ============================================================================
# PREDIKSI
# ============================================================================
elif menu == "🔍 Prediksi":
    st.header("🔍 Deteksi SMS Spam")

    # Example buttons
    col1, col2, col3 = st.columns(3)

    # Initialize session state
    if 'sms_text' not in st.session_state:
        st.session_state.sms_text = ""

    with col1:
        if st.button("✅ Contoh HAM", use_container_width=True):
            st.session_state.sms_text = "Halo kak, gimana kabarnya? Besok kita jadi ketemuan kan?"

    with col2:
        if st.button("📢 Contoh PROMOSI", use_container_width=True):
            st.session_state.sms_text = "Paket Flash 3GB hanya 20rb. Aktifkan di *363#"

    with col3:
        if st.button("🚨 Contoh SPAM", use_container_width=True):
            st.session_state.sms_text = "GRATIS pulsa 100rb! Tekan *123# sekarang juga"

    # Input
    sms_text = st.text_area(
        "Teks SMS",
        value=st.session_state.sms_text,
        height=150,
        placeholder="Masukkan SMS yang ingin dianalisis...",
        key="sms_input"
    )

    if create_prediction_button("🔮 Analisis SMS"):
        if not sms_text.strip():
            st.warning("⚠️ Mohon masukkan teks SMS terlebih dahulu!")
        else:
            # Predict
            sms_tfidf = vectorizer.transform([sms_text])
            prediksi = model.predict(sms_tfidf)[0]
            probabilitas = model.predict_proba(sms_tfidf)[0]

            pred_idx = np.where(model.classes_ == prediksi)[0][0]
            confidence = probabilitas[pred_idx]

            # Display result
            st.markdown("---")
            st.subheader("📊 Hasil Analisis")

            info = LABEL_INFO[prediksi]

            if prediksi == 0:
                st.success(f"### {info['emoji']} {info['name']}")
            elif prediksi == 1:
                st.info(f"### {info['emoji']} {info['name']}")
            else:
                st.error(f"### {info['emoji']} {info['name']}")

            col1, col2 = st.columns([3, 1])

            with col1:
                st.write(f"**Deskripsi:** {info['desc']}")
                st.progress(confidence)
                st.write(f"**Confidence:** {confidence:.1%}")

            with col2:
                st.metric("Label", prediksi)

            # Probabilities
            st.markdown("---")
            st.subheader("📈 Probabilitas Detail")

            for label, prob in zip(model.classes_, probabilitas):
                info_label = LABEL_INFO[label]
                st.write(f"{info_label['emoji']} **{info_label['name']}**: {prob:.1%}")

            # Recommendations
            st.markdown("---")
            st.subheader("💡 Rekomendasi")

            if prediksi == 0:
                st.success("✅ **SMS Aman** - Terdeteksi sebagai pesan normal")
            elif prediksi == 1:
                st.info("📢 **SMS Promosi** - Periksa kredibilitas sebelum klik link")
            else:
                st.error("""
                🚨 **PERHATIAN: Potensi Spam**
                - JANGAN klik link mencurigakan
                - JANGAN berikan info pribadi (PIN, password, OTP)
                - Block nomor pengirim
                """)

# ============================================================================
# MODEL INFO
# ============================================================================
elif menu == "📈 Model Info":
    st.header("📈 Informasi Model")

    st.subheader("🤖 Spesifikasi Model")

    col1, col2 = st.columns(2)

    with col1:
        st.write("""
        **Algoritma:** K-Nearest Neighbors

        **Parameter:**
        - K = 5 tetangga terdekat
        - Distance: Cosine
        - TF-IDF features: 500
        - N-gram: (1, 2)
        """)

    with col2:
        st.write("""
        **Performance:**
        - Accuracy: 89.08%
        - CV Score: 90.70%
        - Train/Test: 914 / 229
        """)

    st.markdown("---")
    st.subheader("📊 Distribusi Data")

    df = pd.read_csv('dataset/02_sms_spam.csv')
    label_counts = df['label'].value_counts()
    label_names = {0: 'Ham', 1: 'Promosi', 2: 'Spam'}

    col1, col2 = st.columns(2)

    with col1:
        st.write("**Jumlah per Kelas:**")
        for label, count in label_counts.items():
            name = label_names[label]
            st.write(f"- {name}: {count} ({count/len(df)*100:.1f}%)")

    with col2:
        chart_data = pd.DataFrame({
            'Kategori': [label_names[l] for l in label_counts.index],
            'Jumlah': label_counts.values
        }).set_index('Kategori')
        st.bar_chart(chart_data)

    st.markdown("---")
    st.subheader("🎯 Cara Penggunaan")
    st.write("""
    1. Pilih menu **Prediksi**
    2. Pilih contoh SMS atau ketik sendiri
    3. Klik **Analisis SMS**
    4. Lihat hasil dan rekomendasi
    """)

    st.info("💡 Model menggunakan TF-IDF untuk ekstraksi fitur dari teks SMS.")

display_footer("SMS Spam Indonesia", "1.1K")
