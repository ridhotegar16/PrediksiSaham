# 📈 Stock Prediction API with Flask and LSTM

Aplikasi ini adalah API berbasis Flask yang menyediakan berbagai layanan analisis saham seperti:
- Prediksi harga saham menggunakan model LSTM
- Pengambilan data historis
- Perhitungan indikator teknikal (SMA)
- Analisis sentimen (dummy)
- Penilaian risiko (volatilitas)

Dilengkapi dengan frontend HTML sederhana yang memungkinkan pengguna untuk berinteraksi dengan API secara langsung.

---

## 🚀 Fitur Utama

- ✅ **Prediksi Harga Saham** menggunakan LSTM
- ✅ **Data Historis** dari Yahoo Finance
- ✅ **Indikator Teknikal** (SMA 20 hari)
- ✅ **Analisis Sentimen** sederhana
- ✅ **Pengukuran Risiko** berbasis volatilitas
- ✅ **Antarmuka Web Interaktif** via `index.html`

---

## 🧠 Teknologi yang Digunakan

- Python, Flask, Flask-CORS
- TensorFlow + Keras (LSTM)
- Scikit-learn
- Yahoo Finance (yfinance)
- HTML + JavaScript frontend

---

---

## ⚙️ Cara Menjalankan (Local)

1. Clone Repo
git clone https://github.com/USERNAME/StockPredictionAPI.git
cd StockPredictionAPI
2.Buat Virtual Environment
python -m venv venv
source venv/bin/activate       # macOS/Linux
venv\Scripts\activate          # Windows
3.Install Dependencies
pip install -r requirements.txt
4.Jalankan Flask API
python app.py



