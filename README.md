# 🦺 Klasifikasi APD (Helm & Kacamata) Menggunakan HOG + PCA: KNN vs Random Forest

Repository ini berisi implementasi proyek **Klasifikasi Alat Pelindung Diri (APD) berupa Helm dan Kacamata** menggunakan pendekatan Pengolahan Citra Digital (PCD). Proyek ini memanfaatkan ekstraksi fitur **HOG (Histogram of Oriented Gradients)** dan reduksi dimensi **PCA (Principal Component Analysis)**, serta membandingkan performa antara metode **K-Nearest Neighbors (KNN)** dan **Random Forest**.

---
## 🌟 Fitur Utama

* **Preprocessing Citra:** Konversi ukuran citra secara seragam (Resize 64x64) dan transformasi ke skala abu-abu (*grayscale*) untuk efisiensi komputasi.
* **Ekstraksi Fitur Robust:** Menggunakan metode HOG untuk menangkap detail tepi, orientasi gradien, dan kontur bentuk dari APD.
* **Reduksi Dimensi Optimal:** Menerapkan PCA menjadi 100 komponen utama untuk meminimalkan risiko *overfitting* sekaligus mempertahankan variansi data yang penting.
* **Analisis Komparatif:** Evaluasi langsung performa akurasi, *precision*, dan *recall* antara algoritma KNN dan Random Forest.
* **Visualisasi Komprehensif:** Menyajikan matriks kebingungan (*confusion matrix*) secara berdampingan serta diagram batang perbandingan akurasi di Jupyter Notebook.

---

## 📊 Hasil Evaluasi & Performa Model

Pengujian dilakukan menggunakan **76 citra uji** (33 Helm & 43 Kacamata) dengan proporsi distribusi yang seimbang. Berikut adalah ringkasan hasil performanya:

| Model / Algoritma | Akurasi | Precision (Helm / Kaca) | Recall (Helm / Kaca) |
| :--- | :---: | :---: | :---: |
| 🟢 **KNN Classifier** ($k=5$) | **87.00%** | **90%** / **85%** | **79%** / **93%** |
| 🔴 **Random Forest** ($100\text{ Trees}$) | **49.00%** | **46%** / **100%** | **100%** / **9%** |

> 📌 **Analisis Singkat:** Model KNN menunjukkan performa yang jauh lebih unggul dan stabil pada ruang metrik koordinat hasil proyeksi PCA dibandingkan dengan algoritma Random Forest (konfigurasi default).

---

## 📁 Struktur Direktori

```text
├── dataset/
│   ├── train/               # Data latih (sub-folder: helm/ dan kacamata/)
│   └── test/                # Data uji (sub-folder: helm/ dan kacamata/)
├── check_knn.py             # Uji cepat model menggunakan subset kecil data
├── test_eval_knn.py         # Skrip evaluasi model pada dataset penuh via terminal
├── make_nb_knn.py           # Pembuat otomatis notebook new_knn.ipynb
├── make_nb_rf.py            # Pembuat otomatis notebook randomforest.ipynb
├── make_nb_compare.py       # Pembuat otomatis notebook pembanding_knn_rf.ipynb
├── new_knn.ipynb            # Jupyter Notebook implementasi KNN
├── randomforest.ipynb       # Jupyter Notebook implementasi Random Forest
├── pembanding_knn_rf.ipynb  # Jupyter Notebook perbandingan visual KNN vs RF
└── README.md                # Dokumentasi utama repositori GitHub
