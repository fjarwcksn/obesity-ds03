# Silakan salin isi kode ini dan simpan sebagai streamlit_app.py di dalam proyek Anda

import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ========================
# Konfigurasi UI
# ========================
st.set_page_config(page_title="Prediksi Obesitas", page_icon="📈", layout="wide")
st.markdown("""
<style>
body {
    background-color: #f8fbfc;
    font-family: 'Segoe UI', sans-serif;
}
h1 {
    color: #0f3460;
    font-weight: bold;
}
.stButton>button {
    background-color: #1b9aaa;
    color: white;
    border-radius: 8px;
    padding: 10px 20px;
    font-weight: bold;
    transition: 0.3s;
}
.stButton>button:hover {
    background-color: #146c81;
}
</style>
""", unsafe_allow_html=True)

st.title("Prediksi Tingkat Obesitas Berdasarkan Gaya Hidup")
st.write("""
Aplikasi ini memprediksi kategori obesitas berdasarkan input kebiasaan dan karakteristik pribadi Anda. 
Silakan isi data berikut:
""")

# ========================
# Form Input Pengguna
# ========================
with st.form("input_form"):
    col1, col2 = st.columns(2)

    with col1:
        age = st.slider("Usia", 10, 100, 25)
        gender = st.selectbox("Jenis Kelamin", ["Male", "Female"])
        height = st.number_input("Tinggi Badan (m)", 1.0, 2.5, step=0.01, value=1.70)
        weight = st.number_input("Berat Badan (kg)", 30.0, 200.0, step=0.1, value=70.0)
        family_history = st.selectbox("Riwayat Keluarga Obesitas", ["yes", "no"])
        favc = st.selectbox("Sering Mengonsumsi Makanan Kalori Tinggi", ["yes", "no"])
        fcvc = st.slider("Frekuensi Konsumsi Sayur (1-3)", 1.0, 3.0, 2.0)
        ncp = st.slider("Jumlah Makan Utama per Hari", 1.0, 4.0, 3.0)

    with col2:
        caec = st.selectbox("Ngemil di luar jam makan", ["no", "Sometimes", "Frequently", "Always"])
        smoke = st.selectbox("Merokok", ["yes", "no"])
        ch2o = st.slider("Konsumsi Air Harian (liter)", 1.0, 3.0, 2.0)
        scc = st.selectbox("Mengurangi Konsumsi Makanan?", ["yes", "no"])
        faf = st.slider("Aktivitas Fisik Mingguan (jam)", 0.0, 5.0, 2.0)
        tue = st.slider("Waktu Layar Harian (jam)", 0.0, 5.0, 2.0)
        calc = st.selectbox("Konsumsi Alkohol", ["no", "Sometimes", "Frequently", "Always"])
        mtrans = st.selectbox("Transportasi", ["Public_Transportation", "Walking", "Automobile", "Motorbike", "Bike"])

    submit = st.form_submit_button("🔍 Prediksi Sekarang")

# ========================
# Proses Prediksi
# ========================
if submit:
    try:
        # Buat DataFrame input pengguna
        user_df = pd.DataFrame({
            'Age': [age],
            'Gender': [gender],
            'Height': [height],
            'Weight': [weight],
            'family_history_with_overweight': [family_history],
            'FAVC': [favc],
            'FCVC': [fcvc],
            'NCP': [ncp],
            'CAEC': [caec],
            'SMOKE': [smoke],
            'CH2O': [ch2o],
            'SCC': [scc],
            'FAF': [faf],
            'TUE': [tue],
            'CALC': [calc],
            'MTRANS': [mtrans]
        })

        # Load model dan tools
        model = pickle.load(open("rf_model.pkl", "rb"))
        scaler = pickle.load(open("scaler.pkl", "rb"))
        expected_columns = pickle.load(open("columns.pkl", "rb"))

        # Encoding input pengguna
        user_encoded = pd.get_dummies(user_df)
        input_data = pd.DataFrame(0, index=[0], columns=expected_columns)
        for col in user_encoded.columns:
            if col in input_data.columns:
                input_data[col] = user_encoded[col]

        input_scaled = scaler.transform(input_data)
        pred = model.predict(input_scaled)[0]
        prob = model.predict_proba(input_scaled)[0].max()

        label_map = ['Insufficient_Weight', 'Normal_Weight', 'Obesity_Type_I', 'Obesity_Type_II',
                     'Obesity_Type_III', 'Overweight_Level_I', 'Overweight_Level_II']

        hasil = label_map[pred]

        if hasil == 'Normal_Weight':
            st.success(f"✅ Anda memiliki berat badan **Normal** dengan keyakinan {prob*100:.1f}%")
        else:
            st.warning(f"⚠️ Anda berada dalam kategori **{hasil.replace('_', ' ')}** dengan keyakinan {prob*100:.1f}%")

    except Exception as e:
        st.error("Terjadi kesalahan saat memproses prediksi. Periksa kembali file model dan dependensi.")
        st.exception(e)
