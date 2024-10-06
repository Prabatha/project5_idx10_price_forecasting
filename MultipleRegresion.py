import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mplfinance as mpf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Fungsi untuk memuat data
def load_data():
    data = pd.read_csv('D:/UNY/semester5/praktik_aplikasi_web/Project/PRAKTIK/BBCA.JK.csv', index_col='Date', parse_dates=True)
    return data

# Fungsi untuk membuat grafik time series
def plot_time_series(data):
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    plt.plot(data[['Open', 'High', 'Low', 'Close']])
    plt.title('Pergerakan Harga Saham BBCA (Time Series)', fontsize=16)
    plt.xlabel('Tanggal', fontsize=14)
    plt.ylabel('Harga', fontsize=14)
    plt.legend(['Open', 'High', 'Low', 'Close'], loc='upper left')
    st.pyplot(plt)
    plt.clf()


# Fungsi untuk membuat grafik volume perdagangan
def plot_volume(data):
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    plt.plot(data['Volume'], color='blue')
    plt.title('Volume Perdagangan Saham BBCA', fontsize=16)
    plt.xlabel('Tanggal', fontsize=14)
    plt.ylabel('Volume', fontsize=14)
    st.pyplot(plt)
    plt.clf()


# Fungsi untuk membuat heatmap korelasi
def plot_heatmap(data):
    plt.figure(figsize=(10, 6))
    sns.heatmap(data[['Open', 'High', 'Low', 'Close', 'Volume']].corr(), annot=True, cmap='coolwarm')
    plt.title('Heatmap Korelasi Atribut Saham', fontsize=16)
    st.pyplot(plt)
    plt.clf()


# Fungsi untuk membuat grafik scatter
def plot_scatter(data):
    plt.figure(figsize=(6, 6))
    sns.scatterplot(data=data, x='High', y='Close')
    plt.title('Hubungan Harga Tertinggi vs Harga Penutupan', fontsize=14)
    st.pyplot(plt)
    plt.clf()

    plt.figure(figsize=(6, 6))
    sns.scatterplot(data=data, x='Low', y='Close')
    plt.title('Hubungan Harga Terendah vs Harga Penutupan', fontsize=14)
    st.pyplot(plt)
    plt.clf()

    plt.figure(figsize=(6, 6))
    sns.scatterplot(data=data, x='Volume', y='Close')
    plt.title('Hubungan Volume Perdagangan vs Harga Penutupan', fontsize=14)
    st.pyplot(plt)
    plt.clf()



# Fungsi untuk membuat box plot yang lebih menarik
def plot_boxplot(data):
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")  # Menggunakan style 'whitegrid' untuk latar belakang

    # Membuat box plot dengan seaborn untuk tampilan yang lebih baik
    sns.boxplot(data=data[['Open', 'High', 'Low', 'Close', 'Volume']], palette="Set2")  # 'Set2' adalah palet warna

    plt.title('Distribusi Harga dan Volume Saham BBCA', fontsize=16)  # Judul plot
    plt.xlabel('Atribut Saham', fontsize=14)  # Label sumbu x
    plt.ylabel('Nilai', fontsize=14)  # Label sumbu y

    st.pyplot(plt)
    plt.clf()


# Fungsi untuk membuat candlestick chart
def plot_candlestick(data):
    mpf.plot(data, type='candle', style='charles', title='Candlestick Chart Saham BBCA', ylabel='Harga', mav=(20, 50), volume=True, show_nontrading=True)
    st.pyplot(plt)
    plt.clf()


# Fungsi untuk menampilkan model regresi dan evaluasinya
def show_regression_model(data):
    features = data[['Open', 'High', 'Low', 'Volume']]
    target = data['Close']
    x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=0)
    model = LinearRegression()
    model.fit(x_train, y_train)
    y_pred_train = model.predict(x_train)
    y_pred_test = model.predict(x_test)
    st.write(f'Train R2 Score: {r2_score(y_train, y_pred_train)}')
    st.write(f'Test R2 Score: {r2_score(y_test, y_pred_test)}')
    return model  # Mengembalikan model untuk digunakan di fungsi prediksi

# Fungsi untuk memprediksi harga saham
def predict_price(model):
    st.header("Prediksi Harga Saham")

    open_price = st.number_input('Harga Pembukaan', min_value=0.0, format='%f')
    high_price = st.number_input('Harga Tertinggi', min_value=0.0, format='%f')
    low_price = st.number_input('Harga Terendah', min_value=0.0, format='%f')
    volume = st.number_input('Volume', min_value=0, format='%d')

    if st.button('Prediksi'):
        predicted_price = model.predict([[open_price, high_price, low_price, volume]])[0]
        st.write(f"Harga Penutupan yang Diprediksi: {predicted_price}")

# Memuat data
data = load_data()

# Menyiapkan layout aplikasi
st.title("Prediksi Harga Saham berbasis Multiple Regression")

# Menambahkan sidebar untuk navigasi
page = st.sidebar.selectbox("Pilih Halaman", ["Grafik Time Series", "Volume Perdagangan", "Heatmap Korelasi", "Grafik Scatter", "Box Plot", "Candlestick Chart", "Model Regresi", "Prediksi Harga Saham"])

# Inisialisasi variabel global untuk model
if 'trained_model' not in st.session_state:
    st.session_state.trained_model = None

if page == "Grafik Time Series":
    st.header("Grafik Time Series")
    plot_time_series(data)
elif page == "Volume Perdagangan":
    st.header("Volume Perdagangan")
    plot_volume(data)
elif page == "Heatmap Korelasi":
    st.header("Heatmap Korelasi")
    plot_heatmap(data)
elif page == "Grafik Scatter":
    st.header("Grafik Scatter")
    plot_scatter(data)
elif page == "Box Plot":
    st.header("Box Plot")
    plot_boxplot(data)
elif page == "Candlestick Chart":
    st.header("Candlestick Chart")
    plot_candlestick(data)
elif page == "Model Regresi":
    st.header("Model Regresi")
    st.session_state.trained_model = show_regression_model(data)
elif page == "Prediksi Harga Saham":
    predict_price(st.session_state.trained_model)