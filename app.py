import streamlit as st
from PIL import Image
import requests
from bs4 import BeautifulSoup
import numpy as np
from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model
from urllib.parse import quote
import os
import time

model = load_model('model_buah_sayur.h5')

labels = {
    0: 'apel', 1: 'pisang', 2: 'bit', 3: 'paprika hijau', 4: 'kubis', 5: 'paprika', 6: 'wortel',
    7: 'kembang kol', 8: 'cabai', 9: 'jagung', 10: 'mentimun', 11: 'terong', 12: 'bawang putih', 13: 'jahe',
    14: 'anggur', 15: 'jalapeno', 16: 'kiwi', 17: 'lemon', 18: 'selada',
    19: 'mangga', 20: 'bawang merah', 21: 'jeruk', 22: 'paprika merah', 23: 'pir', 24: 'kacang polong', 25: 'nanas',
    26: 'delima', 27: 'kentang', 28: 'lobak', 29: 'kedelai', 30: 'bayam', 31: 'jagung',
    32: 'ubi jalar', 33: 'tomat', 34: 'lobak', 35: 'semangka'
}

fruits = ['Apel', 'Pisang', 'Paprika Hijau', 'Cabai', 'Anggur', 'Jalapeno', 'Kiwi', 'Lemon', 'Mangga', 'Jeruk',
          'Paprika Merah', 'Pir', 'Nanas', 'Delima', 'Semangka']

vegetables = ['Bit', 'Kubis', 'Paprika', 'Wortel', 'Kembang Kol', 'Jagung', 'Mentimun', 'Terong', 'Jahe',
              'Selada', 'Bawang Merah', 'Kacang Polong', 'Kentang', 'Lobak', 'Kedelai', 'Bayam',
              'Ubi Jalar', 'Tomat']

def fetch_nutrition_info_improved(prediction):
    """
    Fungsi yang ditingkatkan untuk mengambil informasi nutrisi
    """
    prediction_lower = prediction.lower()
    
    # Debug info
    st.write(f"🔍 Debug: Mencari info untuk '{prediction_lower}'")
    
    # Strategi 2: Coba web scraping sebagai backup
    try:
        st.write("🔍 Debug: Mencoba web scraping...")
        
        # Mapping nama untuk web scraping
        name_mapping = {
            'apel': 'apple',
            'pisang': 'banana', 
            'wortel': 'carrot',
            'tomat': 'tomato',
            'kentang': 'potato',
            'bawang merah': 'onion',
            'jeruk': 'orange',
            'mangga': 'mango',
            'semangka': 'watermelon',
            'anggur': 'grapes'
        }
        
        english_name = name_mapping.get(prediction_lower, prediction_lower)
        
        # Coba beberapa URL
        urls_to_try = [
            f"https://www.nutritionvalue.org/{english_name}-nutrition/",
            f"https://www.nutrition-and-you.com/{english_name}.html",
            f"https://nutritiondata.self.com/facts/fruits-and-fruit-juices/{english_name}"
        ]
        
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        
        for url in urls_to_try:
            try:
                st.write(f"🌐 Debug: Mencoba URL: {url}")
                response = requests.get(url, headers=headers, timeout=8)
                
                if response.status_code == 200:
                    soup = BeautifulSoup(response.text, 'html.parser')
                    
                    # Cari informasi kalori
                    calorie_patterns = ['calorie', 'kcal', 'energy']
                    for pattern in calorie_patterns:
                        elements = soup.find_all(text=lambda text: text and pattern in text.lower())
                        for element in elements[:3]:  # Limit hasil
                            if any(char.isdigit() for char in element):
                                st.write(f"✅ Debug: Ditemukan: {element.strip()}")
                                return f"**Informasi dari web**: {element.strip()}"
                
            except Exception as e:
                st.write(f"❌ Debug: Error pada URL {url}: {str(e)}")
                continue
                
    except Exception as e:
        st.write(f"❌ Debug: Error web scraping: {str(e)}")
    
def prepare_image(img_path):
    img = load_img(img_path, target_size=(224, 224, 3))
    img = img_to_array(img)
    img = img / 255
    img = np.expand_dims(img, [0])
    answer = model.predict(img)
    y_class = answer.argmax(axis=-1)
    y = int(" ".join(str(x) for x in y_class))
    res = labels[y]
    return res

def run():
    st.title("🍎🥬 Klasifikasi Buah dan Sayur")
    st.markdown("Upload gambar buah atau sayuran untuk mendapatkan prediksi dan informasi nutrisi")
    
    # Buat folder upload jika belum ada
    os.makedirs('./upload_images/', exist_ok=True)
    
    img_file = st.file_uploader("Pilih Gambar", type=["jpg", "png", "jpeg"])
    
    if img_file is not None:
        # Tampilkan gambar
        img = Image.open(img_file).resize((250, 250))
        st.image(img, use_column_width=False)
        
        # Simpan gambar
        save_image_path = './upload_images/' + img_file.name
        with open(save_image_path, "wb") as f:
            f.write(img_file.getbuffer())

        # Prediksi
        with st.spinner("🔍 Menganalisis gambar..."):
            result = prepare_image(save_image_path)
            
        hasil_cap = result.capitalize()

        # Tampilkan kategori
        if hasil_cap in vegetables:
            st.info('**🥬 Kategori : Sayuran**')
        else:
            st.info('**🍎 Kategori : Buah**')

        # Tampilkan prediksi
        st.success("**🎯 Prediksi : " + hasil_cap + '**')

        # Ambil informasi nutrisi dengan debugging
        st.subheader("📊 Informasi Nutrisi")
        
        with st.expander("🔧 Debug Info", expanded=True):
            info = fetch_nutrition_info_improved(result)
        
        if info and "tidak ditemukan" not in info.lower():
            st.markdown(info)
        else:
            st.warning("ℹ️ Informasi nutrisi tidak tersedia untuk item ini.")
            
        # Tambahan tips
        with st.expander("💡 Tips Kesehatan"):
            if hasil_cap in fruits:
                st.write("🍎 **Tips Buah**: Konsumsi buah segar lebih baik daripada jus. Makan dengan kulitnya jika memungkinkan untuk mendapat serat maksimal.")
            else:
                st.write("🥬 **Tips Sayuran**: Variasikan warna sayuran untuk mendapat berbagai nutrisi. Metode memasak kukus mempertahankan nutrisi terbaik.")

if __name__ == "__main__":
    run()