import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ==============================
# 🧬 Tampilan UI Modern & Profesional
# ==============================
st.set_page_config(page_title="Prediksi Risiko Obesitas", page_icon="🩺", layout="wide")
st.markdown("""
<style>
body {
    background-color: #f9fbfc;
    font-family: 'Segoe UI', sans-serif;
}
h1, h2, h3 {
    color: #00695c;
    font-weight: 700;
}
.stButton>button {
    background-color: #009688;
    color: white;
    font-weight: bold;
    padding: 8px 20px;
    border-radius: 8px;
}
</style>
""", unsafe_allow_html=True)

st.title("🩺 Prediksi Risiko Obesitas")
st.markdown("#### Form Pemeriksaan Gaya Hidup dan Kesehatan Harian")

st.divider()

# ==============================
# 📥 Form Input Data
# ==============================
with st.form(key="form_prediksi"):
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.slider('Usia', 10, 100, 25)
        gender = st.selectbox('Jenis Kelamin', ['Laki-laki', 'Perempuan'])
        height = st.number_input('Tinggi Badan (meter)', 1.0, 2.5, step=0.01)
        weight = st.number_input('Berat Badan (kg)', 30.0, 200.0, step=0.1)
        family_history = st.selectbox('Riwayat Obesitas Keluarga', ['ya', 'tidak'])

    with col2:
        FAVC = st.selectbox('Konsumsi Makanan Tinggi Kalori?', ['ya', 'tidak'])
        FCVC = st.slider('Frekuensi Konsumsi Sayur (1-3)', 1.0, 3.0, 2.0)
        NCP = st.slider('Jumlah Makan Utama per Hari', 1.0, 4.0, 3.0)
        CAEC = st.selectbox('Ngemil?', ['tidak', 'Kadang-kadang', 'Sering', 'Selalu'])
        CH2O = st.slider('Konsumsi Air Harian (liter)', 1.0, 3.0, 2.0)

    with col3:
        SCC = st.selectbox('Pernah Mengurangi Makan?', ['ya', 'tidak'])
        SMOKE = st.selectbox('Merokok?', ['ya', 'tidak'])
        FAF = st.slider('Aktivitas Fisik per Minggu (jam)', 0.0, 5.0, 2.0)
        TUE = st.slider('Waktu Layar per Hari (jam)', 0.0, 5.0, 2.0)
        CALC = st.selectbox('Konsumsi Alkohol?', ['tidak', 'Kadang-kadang', 'Sering', 'Selalu'])
        MTRANS = st.selectbox('Transportasi Utama', ['Transportasi Umum', 'Jalan Kaki', 'Mobil', 'Motor', 'Sepeda'])

    submit_button = st.form_submit_button("🔍 Prediksi Sekarang")

# ==============================
# 🔮 Proses Prediksi
# ==============================
if submit_button:
    try:
        # Persiapan data input
        user_input = pd.DataFrame({
            'Age': [age],
            'Gender': ["Male" if gender == "Laki-laki" else "Female"],
            'Height': [height],
            'Weight': [weight],
            'family_history_with_overweight': ["yes" if family_history == "ya" else "no"],
            'FAVC': ["yes" if FAVC == "ya" else "no"],
            'FCVC': [FCVC],
            'NCP': [NCP],
            'CAEC': [CAEC],
            'CH2O': [CH2O],
            'SCC': ["yes" if SCC == "ya" else "no"],
            'SMOKE': ["yes" if SMOKE == "ya" else "no"],
            'FAF': [FAF],
            'TUE': [TUE],
            'CALC': [CALC],
            'MTRANS': [MTRANS.replace(" ", "_").replace("Transportasi_Umum", "Public_Transportation")]
        })

        # Load model, scaler, dan kolom
        model = pickle.load(open('rf_model.pkl', 'rb'))
        scaler = pickle.load(open('scaler.pkl', 'rb'))
        expected_columns = pickle.load(open('columns.pkl', 'rb'))

        # Encode & align kolom
        user_input_encoded = pd.get_dummies(user_input)
        input_data = pd.DataFrame(0, index=[0], columns=expected_columns)
        for col in user_input_encoded.columns:
            if col in input_data.columns:
                input_data.at[0, col] = user_input_encoded.at[0, col]

        # Normalisasi & prediksi
        input_scaled = scaler.transform(input_data)
        probs = model.predict_proba(input_scaled)
        prediction = np.argmax(probs)
        confidence = round(np.max(probs) * 100, 1)

        label_map = ['Berat Badan Kurang', 'Normal', 'Obesitas Tipe I', 'Obesitas Tipe II',
                     'Obesitas Tipe III', 'Kelebihan Berat Badan I', 'Kelebihan Berat Badan II']

        hasil = label_map[prediction]

        st.divider()
        if hasil == 'Normal':
            st.success(f"✅ Hasil Prediksi: **{hasil}**")
            st.info(f"📊 Tingkat keyakinan model: {confidence}%")
            st.markdown("🟢 Berat badan Anda termasuk **normal**. Tetap jaga pola makan dan gaya hidup sehat.")
        else:
            st.warning(f"⚠️ Hasil Prediksi: **{hasil}**")
            st.info(f"📊 Tingkat keyakinan model: {confidence}%")
            st.markdown("🔴 Berat badan Anda **tidak dalam kategori normal**. Pertimbangkan pola makan seimbang dan olahraga rutin.")

    except Exception as e:
        st.error("🚫 Terjadi kesalahan saat memproses prediksi.")
        st.exception(e)
