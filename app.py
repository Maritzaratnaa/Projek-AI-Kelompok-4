import streamlit as st
from PIL import Image
import requests
from bs4 import BeautifulSoup
import numpy as np
from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model

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
              'Selada', 'Bawang Merah', 'Kacang Polong', 'Kentang', 'Lobak', 'Kedelai', 'Bayam', 'Jagung',
              'Ubi Jalar', 'Tomat', 'Lobak']


from urllib.parse import quote

def fetch_calories(prediction):
    try:
        formatted = prediction.lower().replace(" ", "-")  # ubah spasi jadi dash
        url = 'https://www.fatsecret.co.id/kalori-gizi/umum/' + quote(formatted)
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) "
                          "Chrome/91.0.4472.124 Safari/537.36"
        }
        req = requests.get(url, headers=headers, timeout=10)
        print("Status Code:", req.status_code)

        soup = BeautifulSoup(req.text, 'html.parser')
        rows = soup.find_all("tr")

        for row in rows:
            cols = row.find_all("td")
            if len(cols) == 2 and "Kalori" in cols[0].text:
                return cols[1].text.strip()

        return "Kalori tidak ditemukan"
    
    except Exception as e:
        st.error("Gagal mengambil data kalori.")
        print("Error:", e)


def prepare_image(img_path):
    img = load_img(img_path, target_size=(224, 224, 3))
    img = img_to_array(img)
    img = img / 255
    img = np.expand_dims(img, [0])
    answer = model.predict(img)
    y_class = answer.argmax(axis=-1)
    print(y_class)
    y = " ".join(str(x) for x in y_class)
    y = int(y)
    res = labels[y]
    print(res)
    return res.capitalize()


def run():
    st.title("Fruits and Vegetable Classification")
    img_file = st.file_uploader("Choose an Image", type=["jpg", "png"])
    if img_file is not None:
        img = Image.open(img_file).resize((250, 250))
        st.image(img, use_column_width=False)
        save_image_path = './upload_images/' + img_file.name
        with open(save_image_path, "wb") as f:
            f.write(img_file.getbuffer())

        if img_file is not None:
            result = prepare_image(save_image_path)
            if result in vegetables:
                st.info('**Category : Vegetables**')
            else:
                st.info('**Category : Fruit**')
            st.success("**Predicted : " + result + '**')
            cal = fetch_calories(result)
            if cal:
                st.warning('**' + cal + '(100 grams)**')

run()