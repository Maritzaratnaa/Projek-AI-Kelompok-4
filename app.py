import streamlit as st
from PIL import Image
import requests
import numpy as np
import pandas as pd
from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model

# Load model
model = load_model('model_buah_sayur.h5')

# Label mapping
labels = {
    0: 'apple', 1: 'banana', 2: 'beetroot', 3: 'bell pepper', 4: 'cabbage', 5: 'capsicum', 6: 'carrot',
    7: 'cauliflower', 8: 'chilli pepper', 9: 'corn', 10: 'cucumber', 11: 'eggplant', 12: 'garlic', 13: 'ginger',
    14: 'grapes', 15: 'jalepeno', 16: 'kiwi', 17: 'lemon', 18: 'lettuce', 19: 'mango', 20: 'onion', 21: 'orange',
    22: 'paprika', 23: 'pear', 24: 'peas', 25: 'pineapple', 26: 'pomegranate', 27: 'potato', 28: 'raddish',
    29: 'soy beans', 30: 'spinach', 31: 'sweetcorn', 32: 'sweetpotato', 33: 'tomato', 34: 'turnip', 35: 'watermelon'
}

fruits = ['Apple', 'Banana', 'Bello Pepper', 'Chilli Pepper', 'Grapes', 'Jalepeno', 'Kiwi', 'Lemon', 'Mango', 'Orange',
          'Paprika', 'Pear', 'Pineapple', 'Pomegranate', 'Watermelon']
vegetables = ['Beetroot', 'Cabbage', 'Capsicum', 'Carrot', 'Cauliflower', 'Corn', 'Cucumber', 'Eggplant', 'Ginger',
              'Lettuce', 'Onion', 'Peas', 'Potato', 'Raddish', 'Soy Beans', 'Spinach', 'Sweetcorn', 'Sweetpotato',
              'Tomato', 'Turnip']

# Fungsi baca CSV gizi
def fetch_nutrition_from_csv(prediction):
    try:
        df = pd.read_csv("nutrisi.csv")
        prediction_lower = prediction.lower()
        row = df[df['nama'] == prediction_lower]

        if row.empty:
            return "Informasi gizi tidak ditemukan."

        return row.iloc[0]
    except Exception as e:
        print(e)
        return "Informasi nutrisi tidak ditemukan."

# Fungsi prediksi gambar
def prepare_image(img_path):
    img = load_img(img_path, target_size=(224, 224, 3))
    img = img_to_array(img) / 255
    img = np.expand_dims(img, axis=0)
    prediction = model.predict(img)
    predicted_class = labels[prediction.argmax()]
    return predicted_class.capitalize()

# UI Streamlit
def run():
    st.markdown("<h1 style='text-align: center; color: green;'>🍎 Fruit & Vegetable Classifier 🍆</h1>", unsafe_allow_html=True)
    st.markdown("Upload gambar buah atau sayuran dan dapatkan prediksi serta informasi gizinya 📊")

    img_file = st.file_uploader("📷 Pilih gambar buah atau sayuran", type=["jpg", "png", "jpeg"])
    if img_file is not None:
        save_image_path = './upload_images/' + img_file.name
        with open(save_image_path, "wb") as f:
            f.write(img_file.getbuffer())

        result = prepare_image(save_image_path)

        col1, col2 = st.columns([1, 2])

        with col1:
            st.image(Image.open(img_file), caption="Gambar yang diunggah", use_column_width=True)

        with col2:
            if result in vegetables:
                st.info("**Kategori : Sayuran 🥦**")
            else:
                st.info("**Kategori : Buah 🍉**")

            st.success(f"**Hasil Prediksi : {result}**")

            # Ambil info gizi
            nutrisi = fetch_nutrition_from_csv(result)
            if isinstance(nutrisi, pd.Series):
                with st.expander("📋 Informasi Gizi per 100 gram"):
                    st.markdown(f"""
                    - **Kalori**: {nutrisi['kalori']} kkal  
                    - **Lemak**: {nutrisi['lemak']} g  
                    - **Protein**: {nutrisi['protein']} g  
                    - **Karbohidrat**: {nutrisi['karbohidrat']} g  
                    - **Vitamin A**: {nutrisi['vitamin_a']} g  
                    - **Vitamin C**: {nutrisi['vitamin_c']} g  
                    """)
            else:
                st.warning(nutrisi)

run()
