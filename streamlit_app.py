import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ==============================
# 🌐 Tampilan UI Profesional Modern
# ==============================
st.set_page_config(page_title="Prediksi Risiko Obesitas", page_icon="🩺", layout="wide")
st.markdown("""
<style>
    html, body {
        background: #f8fbfc;
        font-family: 'Segoe UI', sans-serif;
    }
    h1, h2, h3, h4 {
        color: #003d4d;
        font-weight: 700;
    }
    .block-container {
        padding: 2rem 3rem;
    }
    .stButton>button {
        background-color: #00796b;
        color: white;
        padding: 0.6rem 1.2rem;
        border-radius: 8px;
        font-weight: bold;
        border: none;
    }
    .stSelectbox>div>div {
        background-color: #e0f2f1;
    }
    .stSlider>div {
        background-color: #e0f2f1;
    }
</style>
""", unsafe_allow_html=True)

st.title("🩺 Prediksi Risiko Obesitas")
st.markdown("""
### 📋 Form Pemeriksaan Kesehatan Harian
Isi form berikut dengan informasi dan kebiasaan harian Anda untuk memprediksi tingkat risiko obesitas secara akurat.
""")

# ==============================
# 📥 Form Input Pengguna
# ==============================
with st.form(key='obesity_form'):
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.slider('Usia', 10, 100, 25)
        gender = st.selectbox('Jenis Kelamin', ['Laki-laki', 'Perempuan'])
        height = st.number_input('Tinggi Badan (meter)', 1.0, 2.5, step=0.01)
        weight = st.number_input('Berat Badan (kg)', 30.0, 200.0, step=0.1)
        family_history = st.selectbox('Riwayat Obesitas Keluarga', ['ya', 'tidak'])

    with col2:
        FAVC = st.selectbox('Makanan Tinggi Kalori?', ['ya', 'tidak'])
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

    submit_button = st.form_submit_button(label='🔍 Prediksi Sekarang')

# ==============================
# 🔮 Proses Prediksi
# ==============================
if submit_button:
    try:
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

        # Load model dan tools
        model = pickle.load(open('rf_model.pkl', 'rb'))
        scaler = pickle.load(open('scaler.pkl', 'rb'))
        expected_columns = pickle.load(open('columns.pkl', 'rb'))

        # Preprocessing input user
        user_input_encoded = pd.get_dummies(user_input)
        input_data = pd.DataFrame(0, index=[0], columns=expected_columns)
        for col in user_input_encoded.columns:
            if col in input_data.columns:
                input_data.at[0, col] = user_input_encoded.at[0, col]

        input_scaled = scaler.transform(input_data)

        # Prediksi dan confidence
        probs = model.predict_proba(input_scaled)
        prediction = np.argmax(probs)
        confidence = round(np.max(probs) * 100, 1)

        label_map = ['Berat Badan Kurang', 'Normal', 'Obesitas Tipe I', 'Obesitas Tipe II',
                     'Obesitas Tipe III', 'Kelebihan Berat Badan I', 'Kelebihan Berat Badan II']

        hasil = label_map[prediction]

        if hasil == 'Normal':
            st.success(f"✅ Hasil Prediksi: **{hasil}**")
            st.info(f"📈 Tingkat keyakinan model: {confidence}%")
            st.markdown("---")
            st.markdown("💡 Berat badan Anda termasuk normal. Pertahankan pola hidup sehat dan aktif!")
        else:
            st.warning(f"⚠️ Hasil Prediksi: **{hasil}**")
            st.info(f"📉 Tingkat keyakinan model: {confidence}%")
            st.markdown("---")
            st.markdown("❗ Berat badan Anda tidak dalam kategori normal. Pertimbangkan perubahan pola makan, aktivitas fisik, dan konsultasi ke ahli gizi.")

    except Exception as e:
        st.error("🚫 Terjadi kesalahan saat memproses prediksi. Pastikan file model dan tools tersedia di direktori.")
        st.exception(e)
