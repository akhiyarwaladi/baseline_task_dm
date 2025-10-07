"""
Streamlit Dashboard - Deteksi Stunting Balita
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, export_text
import pickle
import os
from app_utils import create_sidebar_menu, display_metrics, display_footer, create_prediction_button

# Page config
st.set_page_config(page_title="Deteksi Stunting Balita", page_icon="👶", layout="wide")

st.title("👶 Deteksi Stunting Balita Indonesia")
st.markdown("**Model KNN untuk klasifikasi status gizi balita**")

# Sidebar menu
menu = create_sidebar_menu()

# Load or train model
@st.cache_resource
def load_model():
    model_path = 'models/01_model_stunting.pkl'

    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            return pickle.load(f)

    df = pd.read_csv('dataset/01_stunting_balita.csv')
    X = pd.get_dummies(df.drop('Status Gizi', axis=1), drop_first=True)
    y = df['Status Gizi']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale features for KNN
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Train KNN with scaled data
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train_scaled, y_train)

    # Train Decision Tree (for interpretability) - uses unscaled data
    dt = DecisionTreeClassifier(
        random_state=42,
        max_depth=3  # Same as script
    )
    dt.fit(X_train, y_train)

    os.makedirs('models', exist_ok=True)
    with open(model_path, 'wb') as f:
        pickle.dump({
            'model': knn,
            'dt_model': dt,
            'scaler': scaler,  # IMPORTANT: Save scaler!
            'columns': X.columns,
            'feature_names': list(X.columns)
        }, f)

    return {
        'model': knn,
        'dt_model': dt,
        'scaler': scaler,
        'columns': X.columns,
        'feature_names': list(X.columns)
    }

model_data = load_model()
model = model_data['model']
scaler = model_data.get('scaler')  # StandardScaler for KNN
dt_model = model_data.get('dt_model')  # Decision Tree model
feature_names = model_data.get('feature_names', list(model_data['columns']))

# Status info
STATUS_INFO = {
    'normal': {'emoji': '🟢', 'color': 'green', 'desc': 'Status gizi NORMAL'},
    'severely stunted': {'emoji': '🔴', 'color': 'red', 'desc': 'Status gizi SANGAT PENDEK'},
    'stunted': {'emoji': '🟡', 'color': 'orange', 'desc': 'Status gizi PENDEK'},
    'tinggi': {'emoji': '🔵', 'color': 'blue', 'desc': 'Status gizi TINGGI'}
}

# ============================================================================
# HOME
# ============================================================================
if menu == "🏠 Home":
    st.header("Selamat Datang! 👋")

    display_metrics([
        ("Total Data", "120,999 balita"),
        ("Akurasi Model", "99.67%"),
        ("Algoritma", "KNN (K=5)")
    ])

    st.markdown("---")

    st.subheader("📋 Tentang Dataset")
    st.write("""
    Dataset antropometri balita Indonesia untuk deteksi stunting.

    **Kelas Status Gizi:**
    - 🟢 **Normal** (56.0%)
    - 🔴 **Severely Stunted** (16.4%)
    - 🟡 **Stunted** (11.4%)
    - 🔵 **Tinggi** (16.2%)
    """)

    st.subheader("📊 Fitur Input")
    st.write("""
    1. **Umur (bulan)** - Usia balita (0-60)
    2. **Jenis Kelamin** - Laki-laki atau Perempuan
    3. **Tinggi Badan (cm)** - Tinggi badan balita
    """)

# ============================================================================
# PREDIKSI
# ============================================================================
elif menu == "🔍 Prediksi":
    st.header("🔍 Prediksi Status Gizi Balita")

    col1, col2 = st.columns(2)

    with col1:
        umur = st.number_input("Umur (bulan)", 0, 60, 24, help="Usia balita (0-60)")
        jenis_kelamin = st.selectbox("Jenis Kelamin", ["laki-laki", "perempuan"])

    with col2:
        tinggi = st.number_input("Tinggi Badan (cm)", 40.0, 130.0, 85.0, 0.1)

    if create_prediction_button():
        # Prepare and encode input
        input_data = pd.DataFrame({
            'Umur (bulan)': [umur],
            'Jenis Kelamin': [jenis_kelamin],
            'Tinggi Badan (cm)': [tinggi]
        })

        input_encoded = pd.get_dummies(input_data, drop_first=True)

        # Align columns with training data
        for col in model_data['columns']:
            if col not in input_encoded.columns:
                input_encoded[col] = 0
        input_encoded = input_encoded[model_data['columns']]

        # Scale input for KNN (IMPORTANT!)
        if scaler:
            input_scaled = scaler.transform(input_encoded)
        else:
            input_scaled = input_encoded  # Fallback for old models

        # Predict
        prediksi = model.predict(input_scaled)[0]
        probabilitas = model.predict_proba(input_scaled)[0]

        pred_idx = np.where(model.classes_ == prediksi)[0][0]
        confidence = probabilitas[pred_idx]

        # Display result
        st.markdown("---")
        st.subheader("📊 Hasil Prediksi")

        info = STATUS_INFO.get(prediksi, {'emoji': '⚪', 'desc': prediksi})

        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown(f"### {info['emoji']} {info['desc']}")
            st.progress(confidence)
            st.write(f"**Confidence:** {confidence:.1%}")

        with col2:
            st.metric("Status", prediksi.upper())

        # Probabilities
        st.markdown("---")
        st.subheader("📈 Probabilitas Detail")

        for status, prob in zip(model.classes_, probabilitas):
            emoji = STATUS_INFO.get(status, {}).get('emoji', '⚪')
            st.write(f"{emoji} **{status.title()}**: {prob:.1%}")

        # Recommendations
        st.markdown("---")
        st.subheader("💡 Rekomendasi")

        if prediksi in ['severely stunted', 'stunted']:
            st.warning("""
            ⚠️ **Perhatian Khusus Diperlukan**
            - Konsultasi dokter anak/ahli gizi
            - Perbaiki pola makan gizi seimbang
            - Monitor pertumbuhan rutin
            """)
        elif prediksi == 'normal':
            st.success("""
            ✅ **Status Gizi Baik**
            - Pertahankan pola makan bergizi
            - Pemeriksaan kesehatan rutin
            - Pastikan imunisasi lengkap
            """)
        else:
            st.info("ℹ️ **Status Tinggi** - Tetap jaga pola makan sehat")

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
        - Distance: Euclidean
        """)

    with col2:
        st.write("""
        **Performance:**
        - Accuracy: 99.67%
        - CV Score: 99.52%
        - Train/Test: 96,799 / 24,200
        """)

    st.markdown("---")
    st.subheader("📊 Distribusi Data")

    df = pd.read_csv('dataset/01_stunting_balita.csv')
    status_counts = df['Status Gizi'].value_counts()

    col1, col2 = st.columns(2)

    with col1:
        st.write("**Jumlah per Kelas:**")
        for status, count in status_counts.items():
            st.write(f"- {status.title()}: {count:,} ({count/len(df)*100:.1f}%)")

    with col2:
        st.bar_chart(status_counts)

    st.markdown("---")
    st.subheader("🌳 Decision Tree Model (Interpretable)")

    if dt_model:
        st.write("""
        **Model Alternatif:** Decision Tree dengan max_depth=3

        Decision Tree lebih mudah diinterpretasi dibanding KNN, cocok untuk:
        - Memahami faktor-faktor stunting
        - Menjelaskan keputusan ke tenaga medis
        - Mendapatkan insights actionable
        """)

        # Generate tree rules
        tree_rules = export_text(
            dt_model,
            feature_names=feature_names,
            max_depth=3,
            decimals=2,
            show_weights=True
        )

        st.markdown("**📊 Decision Tree Rules:**")

        # Display tree rules - always visible
        st.code(tree_rules, language='text')

        st.info("""
        **💡 Cara Membaca:**
        - Setiap `|---` adalah decision node (keputusan)
        - `class:` menunjukkan status gizi yang diprediksi
        - `weights:` menunjukkan jumlah samples di node tersebut

        **Contoh interpretasi:**
        "Jika Tinggi Badan <= X DAN Umur <= Y → Predicted: Stunted"
        """)
    else:
        st.warning("⚠️ Decision Tree model belum tersedia. Retrain model untuk mendapatkan Decision Tree.")

    st.markdown("---")
    st.subheader("🎯 Cara Penggunaan")
    st.write("""
    1. Pilih menu **Prediksi**
    2. Masukkan data balita (umur, jenis kelamin, tinggi)
    3. Klik **Prediksi**
    4. Lihat hasil dan rekomendasi
    """)

    st.info("💡 Model akurat 99.67%, tetap konsultasi tenaga medis untuk diagnosis pasti.")

display_footer("Stunting Balita Indonesia", "120K")
