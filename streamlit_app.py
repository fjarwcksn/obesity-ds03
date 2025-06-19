import streamlit as st
import pandas as pd
import numpy as np
import joblib # Use joblib for scikit-learn objects, generally more robust than pickle

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
        FAVC = st.selectbox('Sering Konsumsi Makanan Tinggi Kalori?', ['ya', 'tidak'])
        FCVC = st.slider('Frekuensi Konsumsi Sayur (1=Jarang, 2=Kadang, 3=Selalu)', 1.0, 3.0, 2.0)
        NCP = st.slider('Jumlah Makan Utama per Hari', 1.0, 4.0, 3.0)
        CAEC = st.selectbox('Konsumsi Makanan di Antara Waktu Makan Utama (Ngemil)?', ['tidak', 'Kadang-kadang', 'Sering', 'Selalu'])
        CH2O = st.slider('Konsumsi Air Harian (liter)', 1.0, 3.0, 2.0)

    with col3:
        SCC = st.selectbox('Apakah Anda Memantau Konsumsi Kalori Anda?', ['ya', 'tidak'])
        SMOKE = st.selectbox('Merokok?', ['ya', 'tidak'])
        FAF = st.slider('Aktivitas Fisik per Minggu (hari)', 0.0, 7.0, 2.0) # Changed to days/week as per dataset
        TUE = st.slider('Waktu Layar per Hari (jam)', 0.0, 5.0, 2.0)
        CALC = st.selectbox('Konsumsi Alkohol?', ['tidak', 'Kadang-kadang', 'Sering', 'Selalu'])
        MTRANS = st.selectbox('Transportasi Utama', ['Transportasi Umum', 'Jalan Kaki', 'Mobil', 'Motor', 'Sepeda'])

    submit_button = st.form_submit_button(label='🔍 Prediksi Sekarang')

# ==============================
# 🔮 Proses Prediksi
# ==============================
if submit_button:
    try:
        # Load model dan tools
        # Adjust paths if your files are not directly in 'model/'
        model = joblib.load('model/model_rf.pkl')
        scaler = joblib.load('model/scaler.pkl')
        label_encoder_gender = joblib.load('model/label_encoder_gender.pkl') # Assuming you saved this
        # If you used OneHotEncoder for other columns and need specific column order after OHE
        # you might have saved a list of final columns during training
        expected_columns = joblib.load('model/final_columns.pkl') # Assuming you saved this

        user_input = pd.DataFrame({
            'Age': [age],
            'Gender': [gender], # Will be transformed by label_encoder_gender
            'Height': [height],
            'Weight': [weight],
            'family_history_with_overweight': [family_history],
            'FAVC': [FAVC],
            'FCVC': [FCVC],
            'NCP': [NCP],
            'CAEC': [CAEC],
            'CH2O': [CH2O],
            'SCC': [SCC],
            'SMOKE': [SMOKE],
            'FAF': [FAF],
            'TUE': [TUE],
            'CALC': [CALC],
            'MTRANS': [MTRANS]
        })

        # --- Preprocessing Input User ---
        # 1. Map user-friendly labels to dataset labels for MTRANS and others if needed
        # Example: 'Transportasi Umum' -> 'Public_Transportation'
        user_input['MTRANS'] = user_input['MTRANS'].replace({
            'Transportasi Umum': 'Public_Transportation',
            'Jalan Kaki': 'Walking',
            'Mobil': 'Automobile',
            'Motor': 'Motorbike',
            'Sepeda': 'Bike'
        })

        # Map other Indonesian labels to English dataset labels if necessary
        user_input['Gender'] = user_input['Gender'].replace({
            'Laki-laki': 'Male',
            'Perempuan': 'Female'
        })
        user_input['family_history_with_overweight'] = user_input['family_history_with_overweight'].replace({
            'ya': 'yes', 'tidak': 'no'
        })
        user_input['FAVC'] = user_input['FAVC'].replace({
            'ya': 'yes', 'tidak': 'no'
        })
        user_input['SCC'] = user_input['SCC'].replace({
            'ya': 'yes', 'tidak': 'no'
        })
        user_input['SMOKE'] = user_input['SMOKE'].replace({
            'ya': 'yes', 'tidak': 'no'
        })
        user_input['CAEC'] = user_input['CAEC'].replace({
            'tidak': 'no', 'Kadang-kadang': 'Sometimes', 'Sering': 'Frequently', 'Selalu': 'Always'
        })
        user_input['CALC'] = user_input['CALC'].replace({
            'tidak': 'no', 'Kadang-kadang': 'Sometimes', 'Sering': 'Frequently', 'Selalu': 'Always'
        })
        # FCVC and CH2O are already numerical sliders, so no mapping needed if 1-3 range is correct.
        # FAF (Physical Activity Frequency) was in days/week in dataset, adjusted slider max to 7.0

        # 2. Apply Label Encoding for 'Gender'
        user_input['Gender'] = label_encoder_gender.transform(user_input['Gender'])

        # 3. Apply One-Hot Encoding for other categorical features
        # Ensure these match the columns used in training
        categorical_cols_for_ohe = [
            'family_history_with_overweight', 'FAVC', 'CAEC', 'SCC', 'SMOKE', 'CALC', 'MTRANS'
        ]
        user_input_encoded = pd.get_dummies(user_input, columns=categorical_cols_for_ohe, drop_first=False)

        # 4. Align columns with training data (crucial for prediction)
        # Create an empty DataFrame with all expected columns
        final_input_df = pd.DataFrame(0, index=[0], columns=expected_columns)

        # Populate with user's encoded data
        for col in user_input_encoded.columns:
            if col in final_input_df.columns:
                final_input_df.at[0, col] = user_input_encoded.at[0, col]
        
        # Ensure 'FCVC' and 'CH2O' are floats, as they are sliders
        final_input_df['FCVC'] = final_input_df['FCVC'].astype(float)
        final_input_df['CH2O'] = final_input_df['CH2O'].astype(float)

        # 5. Scale numerical features
        # Identify numerical columns that were scaled during training
        numerical_cols_to_scale = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']
        final_input_df[numerical_cols_to_scale] = scaler.transform(final_input_df[numerical_cols_to_scale])

        # Prediksi dan confidence
        probs = model.predict_proba(final_input_df)
        prediction_index = np.argmax(probs)
        confidence = round(np.max(probs) * 100, 1)

        # Define the label map based on your model's output (assuming alphabetical order from LabelEncoder)
        # Verify this order from your Capstone-DS03.ipynb notebook where you encoded the target variable
        label_map = [
            'Insufficient_Weight',      # 0
            'Normal_Weight',            # 1
            'Obesity_Type_I',           # 2
            'Obesity_Type_II',          # 3
            'Obesity_Type_III',         # 4
            'Overweight_Level_I',       # 5
            'Overweight_Level_II'       # 6
        ]
        
        # Adjust the label order if your model's target encoder mapped them differently
        # For example, if your encoder mapped them alphabetically:
        # Insufficient_Weight (0)
        # Normal_Weight (1)
        # Obesity_Type_I (2)
        # Obesity_Type_II (3)
        # Obesity_Type_III (4)
        # Overweight_Level_I (5)
        # Overweight_Level_II (6)
        
        # Convert to more user-friendly Indonesian labels for display
        display_label_map = {
            'Insufficient_Weight': 'Berat Badan Kurang',
            'Normal_Weight': 'Normal',
            'Overweight_Level_I': 'Kelebihan Berat Badan Tingkat I',
            'Overweight_Level_II': 'Kelebihan Berat Badan Tingkat II',
            'Obesity_Type_I': 'Obesitas Tipe I',
            'Obesity_Type_II': 'Obesitas Tipe II',
            'Obesity_Type_III': 'Obesitas Tipe III'
        }

        hasil_dataset_label = label_map[prediction_index]
        hasil_display = display_label_map.get(hasil_dataset_label, "Kategori Tidak Diketahui")

        if hasil_dataset_label == 'Normal_Weight':
            st.success(f"✅ Hasil Prediksi: **{hasil_display}**")
            st.info(f"📈 Tingkat keyakinan model: {confidence}%")
            st.markdown("---")
            st.markdown("💡 Berat badan Anda termasuk normal. Pertahankan pola hidup sehat dan aktif!")
        elif "Overweight" in hasil_dataset_label or "Obesity" in hasil_dataset_label:
            st.warning(f"⚠️ Hasil Prediksi: **{hasil_display}**")
            st.info(f"📉 Tingkat keyakinan model: {confidence}%")
            st.markdown("---")
            st.markdown("❗ Berat badan Anda tidak dalam kategori normal. Pertimbangkan perubahan pola makan, aktivitas fisik, dan konsultasi ke ahli gizi.")
        else: # Insufficient_Weight
            st.info(f"ℹ️ Hasil Prediksi: **{hasil_display}**")
            st.info(f"📉 Tingkat keyakinan model: {confidence}%")
            st.markdown("---")
            st.markdown("💡 Berat badan Anda di bawah normal. Pertimbangkan konsultasi ahli gizi untuk mencapai berat badan ideal.")


    except FileNotFoundError:
        st.error("🚫 File model atau preprocessor tidak ditemukan. Pastikan 'model/model_rf.pkl', 'model/scaler.pkl', 'model/label_encoder_gender.pkl', dan 'model/final_columns.pkl' ada di direktori yang benar.")
    except Exception as e:
        st.error("🚫 Terjadi kesalahan saat memproses prediksi.")
        st.exception(e)
