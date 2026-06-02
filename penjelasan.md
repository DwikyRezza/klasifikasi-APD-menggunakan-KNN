# Penjelasan Proyek Klasifikasi Alat Pelindung Diri (APD) Helm & Kacamata

Dokumen ini menjelaskan secara mendalam tentang arsitektur, metodologi, alur kerja, dan hasil evaluasi proyek klasifikasi citra Alat Pelindung Diri (APD) berupa **Helm** dan **Kacamata**.

---

## 1. Judul Proyek yang Direkomendasikan
Berdasarkan implementasi saat ini, judul yang paling sesuai dan representatif adalah:
> **"Perbandingan Algoritma K-Nearest Neighbors dan Random Forest untuk Klasifikasi Alat Pelindung Diri (Helm dan Kacamata) Berbasis HOG dan PCA"**

---

## 2. Alur Kerja Sistem (Pipeline)
Sistem klasifikasi ini dirancang dengan alur kerja (pipeline) sebagai berikut:

```
[Citra Input (RGB)]
        │
        ▼
[Preprocessing: Resize (64x64) & Grayscale]
        │
        ▼
[Ekstraksi Fitur HOG (Histogram of Oriented Gradients)]
        │
        ▼
[Standard Scaling (Standarisasi Data)]
        │
        ▼
[Reduksi Dimensi PCA (Principal Component Analysis)]
        │
        ▼
┌───────┴─────────────────────────────────┐
▼                                         ▼
[Model KNN Classifier]       [Model Random Forest Classifier]
        │                                 │
        ▼                                 ▼
[Prediksi & Evaluasi]        [Prediksi & Evaluasi]
```

---

## 3. Penjelasan Tahapan Teknis

### A. Preprocessing Citra
1. **Resize (64x64 piksel)**: Semua citra masukan disamakan dimensinya menjadi 64x64 piksel untuk mempercepat komputasi dan memastikan struktur input ke ekstraktor fitur seragam.
2. **Grayscale Conversion**: Citra RGB dikonversi ke skala abu-abu (grayscale) karena ekstraksi fitur bentuk (shape/edge) seperti HOG hanya membutuhkan informasi intensitas kecerahan, bukan informasi warna (karena helm dan kacamata memiliki warna yang sangat bervariasi).

### B. Ekstraksi Fitur HOG (Histogram of Oriented Gradients)
HOG digunakan untuk mengekstrak bentuk objek berdasarkan distribusi arah gradien (orientasi tepi). Parameter yang digunakan:
* `orientations=9`: Membagi arah gradien dalam 9 bin orientasi (0-180 derajat).
* `pixels_per_cell=(8, 8)`: Ukuran sel lokal tempat histogram gradien dihitung.
* `cells_per_block=(2, 2)`: Area normalisasi untuk meminimalkan perubahan pencahayaan.
* Hasil ekstraksi fitur ini menghasilkan matriks fitur berdimensi tinggi (1.764 fitur per citra).

### C. Standardisation & Reduksi Dimensi PCA
* **StandardScaler**: Menyamakan skala dari semua fitur HOG agar memiliki mean = 0 dan variansi = 1. Ini sangat penting untuk KNN karena KNN menggunakan perhitungan jarak Euclidean.
* **PCA (Principal Component Analysis)**: Mengurangi jumlah fitur dari 1.764 menjadi **100 komponen utama (Principal Components)** yang paling mewakili variansi data. Langkah ini bertujuan untuk menghindari *curse of dimensionality* (kutukan dimensi tinggi) dan mencegah overfitting.

### D. Model Klasifikasi

#### 1. K-Nearest Neighbors (KNN)
* **Konsep**: Klasifikasi dilakukan dengan mencari $k$ tetangga terdekat berdasarkan jarak Euclidean terdekat dari titik data baru di ruang fitur hasil PCA.
* **Parameter**: `n_neighbors=5` (menggunakan voting dari 5 tetangga terdekat) dengan metrik jarak Minkowski/Euclidean.
* **Karakteristik**: KNN bekerja sangat baik pada data terkluster dengan batas keputusan yang cukup jelas di ruang dimensi rendah hasil PCA.

#### 2. Random Forest
* **Konsep**: Algoritma berbasis *Ensemble Learning* yang terdiri dari banyak pohon keputusan (Decision Trees). Setiap pohon memberikan prediksi kelas masing-masing, dan kelas dengan perolehan suara terbanyak dipilih sebagai prediksi akhir.
* **Parameter**: `n_estimators=100` (menggunakan 100 pohon keputusan) dengan seed acak `random_state=42`.
* **Karakteristik**: Kuat terhadap pencilan (*outliers*) namun terkadang kesulitan berkinerja optimal langsung pada koordinat proyeksi PCA tanpa tuning kedalaman pohon (*max_depth*).

---

## 4. Evaluasi & Perbandingan Performa
Berdasarkan pengujian pada seluruh dataset (Data Latih: 1.436 Helm & 754 Kacamata; Data Uji: 33 Helm & 43 Kacamata), didapatkan performa sebagai berikut:

### Tabel Performa Model
| Metrik | KNN Classifier | Random Forest Classifier |
| :--- | :---: | :---: |
| **Akurasi Keseluruhan** | **87%** | **49%** |
| **Precision (Helm)** | 90% | 46% |
| **Recall (Helm)** | 79% | 100% |
| **Precision (Kacamata)** | 85% | 100% |
| **Recall (Kacamata)** | 93% | 9% |

### Analisis Hasil: Mengapa KNN Lebih Unggul?
1. **Pengaruh PCA**: KNN bekerja secara alami pada jarak spasial. Setelah fitur HOG disederhanakan oleh PCA menjadi 100 komponen utama yang berdistribusi kontinu, batas keputusan antar-kelas menjadi lebih teratur secara geometris. Jarak Euclidean menjadi representatif untuk mendeteksi kesamaan kelas.
2. **Kelemahan Random Forest pada Kasus Ini**: Random Forest melakukan partisi tegak lurus pada sumbu fitur. Fitur proyeksi PCA (yang merupakan kombinasi linier dari fitur asli) seringkali tidak optimal jika dipisahkan menggunakan pohon keputusan biasa tanpa tuning hyperparameter lebih lanjut (seperti `max_depth` atau `min_samples_split`). Akibatnya, Random Forest mengalami bias yang tinggi (memprediksi hampir semua sampel data uji sebagai "Helm").

---

## 5. Panduan Berkas Proyek

Berikut adalah berkas-berkas yang ada di dalam proyek Anda beserta fungsinya:

### Jupyter Notebook (.ipynb)
* **[new_knn.ipynb](file:///d:/Rezza/Self%20Project/pcd/new_knn.ipynb)**: Berkas implementasi interaktif klasifikasi menggunakan alur HOG + PCA + KNN.
* **[randomforest.ipynb](file:///d:/Rezza/Self%20Project/pcd/randomforest.ipynb)**: Berkas implementasi interaktif klasifikasi menggunakan alur HOG + PCA + Random Forest.
* **[pembanding_knn_rf.ipynb](file:///d:/Rezza/Self%20Project/pcd/pembanding_knn_rf.ipynb)**: Notebook evaluasi komparatif yang menyajikan grafik perbandingan performa, visualisasi matriks kebingungan secara berdampingan, serta grafik batang akurasi.

### Skrip Otomasi Python (.py)
* **[make_nb_knn.py](file:///d:/Rezza/Self%20Project/pcd/make_nb_knn.py)**: Menghasilkan berkas `new_knn.ipynb`.
* **[make_nb_rf.py](file:///d:/Rezza/Self%20Project/pcd/make_nb_rf.py)**: Menghasilkan berkas `randomforest.ipynb`.
* **[make_nb_compare.py](file:///d:/Rezza/Self%20Project/pcd/make_nb_compare.py)**: Menghasilkan berkas `pembanding_knn_rf.ipynb`.
* **[check_knn.py](file:///d:/Rezza/Self%20Project/pcd/check_knn.py)**: Skrip cepat untuk melatih & menguji kedua model pada subset kecil data train (150 gambar per kelas) untuk pengujian instan.
* **[test_eval_knn.py](file:///d:/Rezza/Self%20Project/pcd/test_eval_knn.py)**: Skrip evaluasi performa kedua model pada seluruh dataset melalui terminal/console.
