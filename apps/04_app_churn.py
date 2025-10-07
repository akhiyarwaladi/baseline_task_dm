"""
Streamlit Dashboard - E-Commerce Customer Churn Prediction
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, export_text
import pickle
import os
from app_utils import create_sidebar_menu, display_metrics, display_footer, create_prediction_button

# Page config
st.set_page_config(page_title="Customer Churn Prediction", page_icon="📊", layout="wide")

st.title("📊 E-Commerce Customer Churn Prediction")
st.markdown("**Model KNN untuk prediksi customer churn**")

# Sidebar menu
menu = create_sidebar_menu()

# Load or train model
@st.cache_resource
def load_model():
    model_path = 'models/04_model_churn.pkl'

    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            return pickle.load(f)

    df = pd.read_csv('dataset/04_ecommerce_churn.csv')

    features = ['Tenure', 'PreferredLoginDevice', 'CityTier', 'WarehouseToHome',
                'PreferredPaymentMode', 'Gender', 'HourSpendOnApp', 'NumberOfDeviceRegistered',
                'PreferedOrderCat', 'SatisfactionScore', 'MaritalStatus', 'NumberOfAddress',
                'Complain', 'OrderAmountHikeFromlastYear', 'CouponUsed', 'OrderCount',
                'DaySinceLastOrder', 'CashbackAmount', 'Churn']

    df = df[features].dropna()

    # Encode categoricals
    label_encoders = {}
    for col in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

    X = df.drop('Churn', axis=1)
    y = df['Churn']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Train KNN
    knn = KNeighborsClassifier(n_neighbors=9, metric='euclidean', weights='distance')
    knn.fit(X_train_scaled, y_train)

    # Train Decision Tree (for interpretability)
    dt = DecisionTreeClassifier(
        max_depth=4,  # Limit depth for interpretability
        min_samples_split=50,
        min_samples_leaf=20,
        random_state=42
    )
    dt.fit(X_train, y_train)  # Use unscaled data for tree

    os.makedirs('models', exist_ok=True)
    with open(model_path, 'wb') as f:
        pickle.dump({
            'model': knn,
            'dt_model': dt,
            'scaler': scaler,
            'encoders': label_encoders,
            'columns': X.columns,
            'feature_names': list(X.columns)
        }, f)

    return {
        'model': knn,
        'dt_model': dt,
        'scaler': scaler,
        'encoders': label_encoders,
        'columns': X.columns,
        'feature_names': list(X.columns)
    }

model_data = load_model()
model = model_data['model']
dt_model = model_data.get('dt_model')  # Decision Tree model
scaler = model_data['scaler']
encoders = model_data['encoders']
feature_names = model_data.get('feature_names', list(model_data['columns']))

# HOME
if menu == "🏠 Home":
    st.header("Selamat Datang! 👋")

    display_metrics([
        ("Total Customers", "5,630"),
        ("Akurasi Model", "93.52%"),
        ("ROC-AUC", "0.9705")
    ])

    st.markdown("---")

    st.subheader("📋 Tentang Dataset")
    st.write("""
    Dataset E-Commerce Customer Churn dengan 20 fitur perilaku customer.

    **Target:**
    - **No Churn** (83.2%) - Customer tetap aktif
    - **Churn** (16.8%) - Customer berhenti menggunakan layanan

    **Fitur Penting:**
    - Tenure (lama berlangganan)
    - Complaint (keluhan)
    - Order behavior
    - Payment preferences
    """)

# PREDIKSI
elif menu == "🔍 Prediksi":
    st.header("🔍 Prediksi Customer Churn")

    st.write("Masukkan data customer untuk memprediksi risiko churn:")

    col1, col2, col3 = st.columns(3)

    with col1:
        tenure = st.number_input("Tenure (bulan)", 0, 60, 12)
        complain = st.selectbox("Pernah Komplain?", [0, 1], format_func=lambda x: "Ya" if x else "Tidak")
        satisfaction = st.slider("Satisfaction Score", 1, 5, 3)
        city_tier = st.selectbox("City Tier", [1, 2, 3])

    with col2:
        warehouse_to_home = st.number_input("Jarak Warehouse ke Rumah (km)", 5, 130, 15)
        hour_on_app = st.number_input("Jam per Hari di App", 0, 5, 2)
        num_devices = st.number_input("Jumlah Device Terdaftar", 1, 6, 2)
        num_address = st.number_input("Jumlah Alamat", 1, 22, 2)

    with col3:
        order_count = st.number_input("Total Pesanan", 1, 16, 5)
        cashback = st.number_input("Total Cashback (Rp)", 0, 350, 100)
        days_last_order = st.number_input("Hari Sejak Pesanan Terakhir", 0, 46, 5)
        coupon_used = st.number_input("Kupon Digunakan", 0, 16, 2)

    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        marital_status = st.selectbox("Status Pernikahan", ["Single", "Married", "Divorced"])
        login_device = st.selectbox("Login Device", ["Mobile Phone", "Computer"])

    with col2:
        payment_mode = st.selectbox("Payment Mode", ["Credit Card", "Debit Card", "E wallet", "Cash on Delivery"])
        order_cat = st.selectbox("Kategori Pesanan Favorit", ["Laptop & Accessory", "Mobile Phone", "Fashion", "Grocery", "Others"])
        order_hike = st.slider("Kenaikan Order (%)", 10, 30, 15)

    if create_prediction_button():
        # Prepare input
        input_data = pd.DataFrame({
            'Tenure': [tenure],
            'PreferredLoginDevice': [login_device],
            'CityTier': [city_tier],
            'WarehouseToHome': [warehouse_to_home],
            'PreferredPaymentMode': [payment_mode],
            'Gender': [gender],
            'HourSpendOnApp': [hour_on_app],
            'NumberOfDeviceRegistered': [num_devices],
            'PreferedOrderCat': [order_cat],
            'SatisfactionScore': [satisfaction],
            'MaritalStatus': [marital_status],
            'NumberOfAddress': [num_address],
            'Complain': [complain],
            'OrderAmountHikeFromlastYear': [order_hike],
            'CouponUsed': [coupon_used],
            'OrderCount': [order_count],
            'DaySinceLastOrder': [days_last_order],
            'CashbackAmount': [cashback]
        })

        # Encode
        for col, encoder in encoders.items():
            if col in input_data.columns:
                input_data[col] = encoder.transform(input_data[col])

        # Align columns
        input_data = input_data[model_data['columns']]

        # Scale and predict
        input_scaled = scaler.transform(input_data)
        prediksi = model.predict(input_scaled)[0]
        probabilitas = model.predict_proba(input_scaled)[0]

        churn_prob = probabilitas[1]

        # Display result
        st.markdown("---")
        st.subheader("📊 Hasil Prediksi")

        if churn_prob > 0.7:
            st.error(f"### 🚨 HIGH RISK - Churn Probability: {churn_prob:.1%}")
            risk_level = "HIGH"
        elif churn_prob > 0.4:
            st.warning(f"### ⚠️ MEDIUM RISK - Churn Probability: {churn_prob:.1%}")
            risk_level = "MEDIUM"
        else:
            st.success(f"### ✅ LOW RISK - Churn Probability: {churn_prob:.1%}")
            risk_level = "LOW"

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Prediksi", "Churn" if prediksi == 1 else "No Churn")

        with col2:
            st.metric("Risk Level", risk_level)

        with col3:
            st.metric("Confidence", f"{max(probabilitas):.1%}")

        # Probabilities
        st.markdown("---")
        st.subheader("📈 Probabilitas Detail")

        st.write(f"✅ **No Churn**: {probabilitas[0]:.1%}")
        st.write(f"🚨 **Churn**: {probabilitas[1]:.1%}")

        # Recommendations
        st.markdown("---")
        st.subheader("💡 Rekomendasi Aksi")

        if churn_prob > 0.7:
            st.error("""
            🚨 **URGENT ACTION REQUIRED**
            - Hubungi customer segera
            - Tawarkan special discount/cashback
            - Investigasi keluhan
            - Follow up intensif
            """)
        elif churn_prob > 0.4:
            st.warning("""
            ⚠️ **PROACTIVE ENGAGEMENT**
            - Kirim retention campaign
            - Tawarkan loyalty program
            - Tingkatkan customer service
            - Monitor aktivitas
            """)
        else:
            st.success("""
            ✅ **MAINTAIN RELATIONSHIP**
            - Customer engagement normal
            - Terus berikan value
            - Maintain service quality
            """)

# MODEL INFO
elif menu == "📈 Model Info":
    st.header("📈 Informasi Model")

    st.subheader("🤖 Spesifikasi Model")

    col1, col2 = st.columns(2)

    with col1:
        st.write("""
        **Algoritma:** K-Nearest Neighbors

        **Parameter:**
        - K = 9 tetangga terdekat
        - Distance: Euclidean (weighted)
        - Preprocessing: StandardScaler
        """)

    with col2:
        st.write("""
        **Performance:**
        - Accuracy: 93.52%
        - ROC-AUC: 0.9705
        - CV Score: 91.55%
        - Train/Test: 4,504 / 1,126
        """)

    st.markdown("---")
    st.subheader("📊 Distribusi Churn")

    df = pd.read_csv('dataset/04_ecommerce_churn.csv')
    churn_counts = df['Churn'].value_counts()

    col1, col2 = st.columns(2)

    with col1:
        st.write("**Jumlah per Kelas:**")
        for churn, count in churn_counts.items():
            label = "No Churn" if churn == 0 else "Churn"
            st.write(f"- {label}: {count:,} ({count/len(df)*100:.1f}%)")

    with col2:
        chart_data = pd.DataFrame({
            'Status': ['No Churn', 'Churn'],
            'Jumlah': [churn_counts.get(0, 0), churn_counts.get(1, 0)]
        }).set_index('Status')
        st.bar_chart(chart_data)

    st.markdown("---")
    st.subheader("🌳 Decision Tree Model (Interpretable)")

    if dt_model:
        st.write("""
        **Model Alternatif:** Decision Tree dengan max_depth=4

        Decision Tree lebih mudah diinterpretasi dibanding KNN, cocok untuk:
        - Memahami faktor-faktor churn
        - Menjelaskan keputusan ke business team
        - Mendapatkan insights actionable
        """)

        # Generate tree rules
        tree_rules = export_text(
            dt_model,
            feature_names=feature_names,
            max_depth=4,
            decimals=2,
            show_weights=True
        )

        st.markdown("**📊 Decision Tree Rules:**")

        # Display tree rules - always visible
        st.code(tree_rules, language='text')

        st.info("""
        **💡 Cara Membaca:**
        - Setiap `|---` adalah decision node (keputusan)
        - Angka di akhir `class: 0` = No Churn, `class: 1` = Churn
        - `weights:` menunjukkan jumlah samples di node tersebut

        **Contoh interpretasi:**
        "Jika Complain = 1 DAN Tenure <= 5 bulan → HIGH RISK CHURN"
        """)
    else:
        st.warning("⚠️ Decision Tree model belum tersedia. Retrain model untuk mendapatkan Decision Tree.")

    st.markdown("---")
    st.subheader("🎯 Top 5 Fitur Penting")

    st.write("""
    1. **Tenure** - Lama berlangganan
    2. **Complain** - Riwayat keluhan
    3. **CashbackAmount** - Total cashback
    4. **DaySinceLastOrder** - Aktivitas terakhir
    5. **SatisfactionScore** - Kepuasan customer
    """)

    st.info("💡 Focus on fitur-fitur ini untuk retention strategy yang efektif.")

display_footer("E-Commerce Customer Churn", "5.6K")
