import streamlit as st
from PIL import Image
import requests
from bs4 import BeautifulSoup
import numpy as np
from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model
from urllib.parse import quote

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


def fetch_nutrition_info(prediction):
    try:
        nama_bahan = quote(prediction.lower().replace(" ", "-"))
        url = f"https://www.fatsecret.co.id/kalori-gizi/umum/{nama_bahan}"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) "
                          "Chrome/91.0.4472.124 Safari/537.36"
        }
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")

        table = soup.find("table", class_="generic searchResult")
        if table is None:
            return "Informasi gizi tidak ditemukan."

        result = ""
        rows = table.find_all("tr")
        for row in rows:
            cols = row.find_all("td")
            if len(cols) == 2:
                label = cols[0].text.strip()
                value = cols[1].text.strip()
                result += f"- {label}: {value}\n"

        return result.strip()

    except Exception as e:
        print("Error:", e)
        return "Gagal mengambil data gizi."


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
    st.title("Klasifikasi Buah dan Sayur")
    img_file = st.file_uploader("Pilih Gambar", type=["jpg", "png"])
    if img_file is not None:
        img = Image.open(img_file).resize((250, 250))
        st.image(img, use_column_width=False)
        save_image_path = './upload_images/' + img_file.name
        with open(save_image_path, "wb") as f:
            f.write(img_file.getbuffer())

        result = prepare_image(save_image_path)
        hasil_cap = result.capitalize()

        if hasil_cap in vegetables:
            st.info('**Kategori : Sayuran**')
        else:
            st.info('**Kategori : Buah**')

        st.success("**Prediksi : " + hasil_cap + '**')

        info = fetch_nutrition_info(result)
        if info:
            st.info("**Informasi Gizi (per 100 gram):**")
            st.text(info)


run()
