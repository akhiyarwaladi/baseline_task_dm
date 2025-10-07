"""
Streamlit Dashboard - Tokopedia Review Emotion Classification
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
st.set_page_config(page_title="Emotion Classification", page_icon="😊", layout="wide")

st.title("😊 Tokopedia Review Emotion Classification")
st.markdown("**Model KNN untuk deteksi emosi dari review produk**")

# Sidebar menu
menu = create_sidebar_menu()

# Load or train model
@st.cache_resource
def load_model():
    model_path = 'models/03_model_emotion.pkl'

    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            return pickle.load(f)

    df = pd.read_csv('dataset/03_tokopedia_emotion.csv').dropna(subset=['Customer Review', 'Emotion'])
    X, y = df['Customer Review'], df['Emotion']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2), min_df=2)
    X_train_tfidf = vectorizer.fit_transform(X_train)

    knn = KNeighborsClassifier(n_neighbors=7, metric='cosine', weights='distance')
    knn.fit(X_train_tfidf, y_train)

    os.makedirs('models', exist_ok=True)
    with open(model_path, 'wb') as f:
        pickle.dump({'model': knn, 'vectorizer': vectorizer}, f)

    return {'model': knn, 'vectorizer': vectorizer}

model_data = load_model()
model, vectorizer = model_data['model'], model_data['vectorizer']

# Emotion info
EMOTION_INFO = {
    'Happy': {'emoji': '😊', 'desc': 'Senang/Puas'},
    'Sadness': {'emoji': '😢', 'desc': 'Sedih/Kecewa'},
    'Anger': {'emoji': '😠', 'desc': 'Marah'},
    'Fear': {'emoji': '😨', 'desc': 'Takut/Khawatir'},
    'Love': {'emoji': '❤️', 'desc': 'Cinta/Sangat Suka'}
}

# HOME
if menu == "🏠 Home":
    st.header("Selamat Datang! 👋")

    display_metrics([
        ("Total Reviews", "5,400"),
        ("Akurasi Model", "53.15%"),
        ("Algoritma", "KNN (K=7)")
    ])

    st.markdown("---")

    st.subheader("📋 Tentang PRDECT-ID Dataset")
    st.write("""
    Dataset pertama di Indonesia untuk emotion classification pada review e-commerce.

    **5 Kategori Emosi:**
    - 😊 **Happy** - 32.8%
    - 😢 **Sadness** - 22.3%
    - 😨 **Fear** - 17.0%
    - ❤️ **Love** - 15.0%
    - 😠 **Anger** - 12.9%
    """)

    st.markdown("---")
    st.subheader("📝 Contoh Review per Kategori Emosi")

    col1, col2 = st.columns(2)

    with col1:
        st.success("**😊 HAPPY (Senang/Puas)**")
        st.write("_'Barang bagus banget! Seller ramah, pengiriman cepat. Sangat puas!'_")
        st.write("_'Kualitas produk mantap, sesuai ekspektasi. Recommend!'_")

        st.info("**😢 SADNESS (Sedih/Kecewa)**")
        st.write("_'Kecewa banget, barang tidak sesuai deskripsi.'_")
        st.write("_'Sayang sekali, warnanya beda dari foto. Mengecewakan.'_")

        st.error("**😠 ANGER (Marah)**")
        st.write("_'Paket sudah diterima tapi rusak. Tolong dikembalikan uang saya!'_")
        st.write("_'Sangat mengecewakan! Barang palsu! Penjual tidak bertanggung jawab!'_")

    with col2:
        st.warning("**😨 FEAR (Takut/Khawatir)**")
        st.write("_'Takut barang tidak sampai, tapi ternyata aman.'_")
        st.write("_'Khawatir kualitasnya jelek, tapi untungnya bagus.'_")

        st.markdown("**❤️ LOVE (Cinta/Sangat Suka)**")
        st.write("_'Produk ini amazing! Saya suka sekali. Recommended!'_")
        st.write("_'Love it! Bakal beli lagi. The best seller ever!'_")

# PREDIKSI
elif menu == "🔍 Prediksi":
    st.header("🔍 Deteksi Emosi Review")

    # Example buttons
    col1, col2, col3, col4, col5 = st.columns(5)

    # Initialize session state
    if 'review_text' not in st.session_state:
        st.session_state.review_text = ""

    with col1:
        if st.button("😊 Happy", use_container_width=True):
            st.session_state.review_text = "Barang bagus banget! Seller ramah, pengiriman cepat. Sangat puas!"

    with col2:
        if st.button("😢 Sad", use_container_width=True):
            st.session_state.review_text = "Kecewa banget, barang tidak sesuai deskripsi."

    with col3:
        if st.button("😠 Angry", use_container_width=True):
            st.session_state.review_text = "Paket sudah diterima tapi rusak. Tolong dikembalikan uang saya!"

    with col4:
        if st.button("😨 Fear", use_container_width=True):
            st.session_state.review_text = "Takut barang tidak sampai, tapi ternyata aman."

    with col5:
        if st.button("❤️ Love", use_container_width=True):
            st.session_state.review_text = "Produk ini amazing! Saya suka sekali. Recommended!"

    # Input
    review_text = st.text_area(
        "Teks Review",
        value=st.session_state.review_text,
        height=150,
        placeholder="Masukkan review produk...",
        key="review_input"
    )

    if create_prediction_button("🔮 Analisis Emosi"):
        if not review_text.strip():
            st.warning("⚠️ Mohon masukkan review terlebih dahulu!")
        else:
            # Predict
            review_tfidf = vectorizer.transform([review_text])
            prediksi = model.predict(review_tfidf)[0]
            probabilitas = model.predict_proba(review_tfidf)[0]

            pred_idx = list(model.classes_).index(prediksi)
            confidence = probabilitas[pred_idx]

            # Display result
            st.markdown("---")
            st.subheader("📊 Hasil Analisis")

            info = EMOTION_INFO[prediksi]
            st.markdown(f"### {info['emoji']} **{prediksi}** ({info['desc']})")

            col1, col2 = st.columns([3, 1])

            with col1:
                st.progress(confidence)
                st.write(f"**Confidence:** {confidence:.1%}")

            with col2:
                st.metric("Emosi", prediksi)

            # Probabilities
            st.markdown("---")
            st.subheader("📈 Probabilitas Detail")

            for emotion, prob in zip(model.classes_, probabilitas):
                emoji = EMOTION_INFO[emotion]['emoji']
                st.write(f"{emoji} **{emotion}**: {prob:.1%}")

            # Recommendations
            st.markdown("---")
            st.subheader("💡 Rekomendasi untuk Seller")

            if prediksi == 'Anger':
                st.error("😠 **Prioritas Tinggi** - Segera hubungi customer, tawarkan solusi")
            elif prediksi == 'Sadness':
                st.warning("😢 **Perlu Perhatian** - Follow up dan perbaiki masalah")
            elif prediksi in ['Happy', 'Love']:
                st.success("😊 **Customer Puas** - Pertahankan kualitas layanan")
            else:
                st.info("😨 **Monitor** - Pastikan customer merasa aman")

# MODEL INFO
elif menu == "📈 Model Info":
    st.header("📈 Informasi Model")

    st.subheader("🤖 Spesifikasi Model")

    col1, col2 = st.columns(2)

    with col1:
        st.write("""
        **Algoritma:** K-Nearest Neighbors

        **Parameter:**
        - K = 7 tetangga terdekat
        - Distance: Cosine (weighted)
        - TF-IDF features: 1000
        - N-gram: (1, 2)
        """)

    with col2:
        st.write("""
        **Performance:**
        - Accuracy: 53.15%
        - CV Score: 52.80%
        - Train/Test: 4,320 / 1,080
        """)

    st.markdown("---")
    st.subheader("📊 Distribusi Data")

    df = pd.read_csv('dataset/03_tokopedia_emotion.csv')
    emotion_counts = df['Emotion'].value_counts()

    col1, col2 = st.columns(2)

    with col1:
        st.write("**Jumlah per Emosi:**")
        for emotion, count in emotion_counts.items():
            emoji = EMOTION_INFO.get(emotion, {}).get('emoji', '')
            st.write(f"- {emoji} {emotion}: {count} ({count/len(df)*100:.1f}%)")

    with col2:
        st.bar_chart(emotion_counts)

    st.markdown("---")
    st.subheader("🎯 Cara Penggunaan")
    st.write("""
    1. Pilih menu **Prediksi**
    2. Pilih contoh atau ketik review
    3. Klik **Analisis Emosi**
    4. Lihat hasil dan rekomendasi
    """)

    st.info("💡 Model ini membantu seller memahami sentiment customer untuk meningkatkan layanan.")

display_footer("PRDECT-ID Tokopedia Reviews", "5.4K")
