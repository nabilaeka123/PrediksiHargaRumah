import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pickle

# Load dataset
image = Image.open("house.jpg")
data = pd.read_csv("Data Clean.csv")

# Sidebar menu
menu = st.sidebar.selectbox("Pilih Halaman", ["Deskripsi", "Dataset", "Grafik", "Prediksi"])


# Fungsi untuk halaman Deskripsi
def show_deskripsi():
    st.title("Aplikasi Prediksi Harga Rumah")
    st.image(image, use_column_width=True)
    st.write("""
        <div style='text-align: justify;'>
        Selamat datang di aplikasi prediksi harga rumah. Aplikasi ini menggunakan model
        Machine Learning untuk memprediksi harga rumah berdasarkan parameter tertentu seperti
        luas bangunan, jumlah kamar mandi, kondisi rumah, dll. Dataset yang digunakan
        merupakan data rumah di suatu wilayah dengan beragam variabel. 
        </div>
    """, unsafe_allow_html=True)


# Fungsi untuk halaman Dataset
def show_dataset():
    st.header("Dataset")
    st.write("Berikut adalah cuplikan data yang digunakan:")
    st.dataframe(data.head(10))
    st.write("Dataset berisi {} baris dan {} kolom.".format(data.shape[0], data.shape[1]))


# Fungsi untuk halaman Grafik
def show_grafik():
    st.header("Grafik")
    tab1, tab2, tab3 = st.tabs(["Luas Bangunan", "Jumlah Kamar Mandi", "Kondisi Rumah"])

    with tab1:
        st.write("Grafik Luas Bangunan")
        st.line_chart(data["sqft_living"])

    with tab2:
        st.write("Grafik Jumlah Kamar Mandi")
        st.bar_chart(data["bathrooms"])

    with tab3:
        st.write("Grafik Kondisi Rumah")
        st.bar_chart(data["condition"])


# Fungsi untuk halaman Prediksi
def show_prediksi():
    st.header("Prediksi Harga Rumah")

    # Input parameters
    sqft_liv = st.slider("Luas Bangunan (sqft)", int(data.sqft_living.min()), int(data.sqft_living.max()),
                         int(data.sqft_living.mean()))
    sqft_abo = st.slider("Luas Atas Tanah (sqft)", int(data.sqft_above.min()), int(data.sqft_above.max()),
                         int(data.sqft_above.mean()))
    bath = st.slider("Jumlah Kamar Mandi", int(data.bathrooms.min()), int(data.bathrooms.max()),
                     int(data.bathrooms.mean()))
    view = st.slider("Pemandangan (Skor)", int(data.view.min()), int(data.view.max()), int(data.view.mean()))
    sqft_bas = st.slider("Luas Basement (sqft)", int(data.sqft_basement.min()), int(data.sqft_basement.max()),
                         int(data.sqft_basement.mean()))
    condition = st.slider("Kondisi Rumah", int(data.condition.min()), int(data.condition.max()),
                          int(data.condition.mean()))

    # Split data
    X = data.drop('price', axis=1)
    y = data['price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=45)

    # Model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict
    predictions = model.predict([[sqft_liv, sqft_abo, bath, view, sqft_bas, condition]])[0]
    errors = np.sqrt(mean_squared_error(y_test, model.predict(X_test)))
    akurasi = r2_score(y_test, model.predict(X_test))

    if st.button("Prediksi"):
        st.subheader("Hasil Prediksi")
        st.write(f"Harga Rumah: USD {int(predictions)}")
        st.write(f"Rentang Prediksi: USD {int(predictions - errors)} - USD {int(predictions + errors)}")
        st.write(f"Akurasi Model: {akurasi:.2f}")


# Menampilkan halaman sesuai pilihan menu
if menu == "Deskripsi":
    show_deskripsi()
elif menu == "Dataset":
    show_dataset()
elif menu == "Grafik":
    show_grafik()
elif menu == "Prediksi":
    show_prediksi()
