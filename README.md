# 🛒 Online Shoppers Intelligence Dashboard

> **End-to-end data analytics & machine learning project** untuk menganalisis perilaku pengunjung e-commerce dan memprediksi konversi pembelian secara real-time.

![Python](https://img.shields.io/badge/Python-3.35-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.54-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.6-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-2.3-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-6.5-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)

---

## 📌 Overview

Project ini merupakan **portfolio data science** yang membangun dashboard interaktif berbasis web menggunakan Python dan Streamlit. Data yang digunakan adalah **Online Shoppers Purchasing Intention Dataset** dari Kaggle, berisi 12.330 sesi pengunjung toko online dengan 18 fitur perilaku.

### 🎯 Tujuan
- Menganalisis pola perilaku pengunjung e-commerce
- Membangun model Machine Learning untuk memprediksi apakah pengunjung akan melakukan pembelian
- Menyajikan hasil analisis dalam dashboard interaktif yang dapat diakses publik

---

## 🖥️ Live Demo

🔗 **[Lihat Dashboard →](https://your-app-link.streamlit.app)**

---

## 📊 Dataset

| Info | Detail |
|------|--------|
| **Sumber** | [Kaggle - Online Shoppers Intention](https://www.kaggle.com/datasets/henrysue/online-shoppers-intention) |
| **Jumlah Baris** | 12.330 sesi pengunjung |
| **Jumlah Fitur** | 18 kolom |
| **Target** | `Revenue` (True/False) |
| **Missing Values** | 0 (bersih) |

### Fitur Utama Dataset
| Kolom | Tipe | Deskripsi |
|-------|------|-----------|
| `Administrative` | int | Jumlah halaman administrasi dikunjungi |
| `ProductRelated` | int | Jumlah halaman produk dikunjungi |
| `BounceRates` | float | Persentase pengunjung yang langsung keluar |
| `ExitRates` | float | Persentase keluar dari halaman tertentu |
| `PageValues` | float | Nilai rata-rata halaman sebelum transaksi |
| `Month` | str | Bulan kunjungan |
| `VisitorType` | str | New Visitor / Returning Visitor / Other |
| `Weekend` | bool | Apakah kunjungan di akhir pekan |
| `Revenue` | bool | **Target** — apakah terjadi pembelian |

---

## ✨ Fitur Dashboard

### 📊 Exploratory Data Analysis
- Distribusi revenue (pie chart interaktif)
- Tren transaksi per bulan
- Perbandingan konversi Weekday vs Weekend
- Analisis Visitor Type vs Revenue
- Scatter plot Bounce Rate vs Exit Rate
- Box plot PageValues per bulan
- Line chart konversi vs Special Day
- Heatmap korelasi antar fitur

### 🤖 Machine Learning
- Perbandingan performa 2 model (Random Forest vs Logistic Regression)
- Feature Importance chart
- Tabel ranking fitur terpenting
- Confusion Matrix dengan breakdown detail

### 🎯 Prediksi Interaktif
- Input data pengunjung secara manual
- Prediksi real-time apakah pengunjung akan membeli
- Menampilkan probabilitas prediksi

### 💡 Business Insights
- 7 insight bisnis otomatis berdasarkan data
- Rekomendasi strategi marketing berdasarkan pola data

### ⚙️ Filter Dinamis
- Filter berdasarkan bulan
- Filter berdasarkan tipe visitor
- Filter Weekday / Weekend
- Download data hasil filter (CSV)

---

## 🤖 Machine Learning

### Model yang Digunakan

| Model | Akurasi | Keterangan |
|-------|---------|------------|
| **Random Forest** | **89.6%** ⭐ | Model terbaik, digunakan untuk prediksi interaktif |
| Logistic Regression | 86.9% | Baseline model dengan scaling |

### Preprocessing
- Label Encoding untuk fitur kategorikal (`Month`, `VisitorType`)
- Standard Scaling untuk Logistic Regression
- Train-test split 80:20 dengan `random_state=42`

### Top 3 Fitur Terpenting (Random Forest)
1. `PageValues` — nilai halaman sebelum transaksi
2. `ExitRates` — tingkat keluar dari halaman
3. `ProductRelated_Duration` — durasi di halaman produk

---

## 🗂️ Struktur Project

```
sales-dashboard/
│
├── app.py                  # Dashboard utama Streamlit
├── eda.py                  # Script Exploratory Data Analysis
├── ml.py                   # Script Machine Learning
├── requirements.txt        # Dependencies
├── online_shoppers_intention.csv  # Dataset
│
└── charts/                 # Output chart dari EDA & ML
    ├── chart_revenue.png
    ├── chart_monthly.png
    ├── chart_visitor.png
    ├── chart_correlation.png
    ├── chart_feature_importance.png
    └── chart_confusion_matrix.png
```

---

## 🚀 Cara Menjalankan Locally

### 1. Clone Repository
```bash
git clone https://github.com/USERNAME/sales-dashboard.git
cd sales-dashboard
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Jalankan Dashboard
```bash
python -m streamlit run app.py
```

### 4. Buka di Browser
```
http://localhost:8501
```

---

## 🛠️ Tech Stack

| Teknologi | Versi | Fungsi |
|-----------|-------|--------|
| Python | 3.13 | Bahasa pemrograman utama |
| Streamlit | 1.54 | Framework web dashboard |
| Pandas | 2.3 | Manipulasi & analisis data |
| NumPy | 2.4 | Komputasi numerik |
| Scikit-Learn | latest | Machine learning models |
| Plotly | 6.5 | Visualisasi interaktif |
| Seaborn | 0.13 | Visualisasi statistik |
| Matplotlib | 3.10 | Visualisasi dasar |

---

## 📈 Key Findings

- Hanya **15.5%** pengunjung yang melakukan pembelian — peluang optimasi konversi masih besar
- **November** adalah bulan dengan transaksi tertinggi
- **Returning Visitor** lebih banyak melakukan pembelian dibanding New Visitor
- **PageValues** adalah fitur paling berpengaruh terhadap keputusan pembelian
- Pengunjung dengan **BounceRate & ExitRate rendah** cenderung lebih banyak membeli



## 📄 License

This project is licensed under the MIT License.

