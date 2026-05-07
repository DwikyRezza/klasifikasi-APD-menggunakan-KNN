import nbformat as nbf

nb = nbf.v4.new_notebook()

text_intro = """\
# Klasifikasi Warna Helm Proyek
Alur pemrosesan: **Citra Asli -> Konversi HSV -> Thresholding Warna (Masking) -> Deteksi Tepi (Sobel/Prewitt) -> Klasifikasi**
"""

code_imports = """\
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from tqdm.notebook import tqdm
from ipywidgets import interact, widgets
from IPython.display import display

# Helper function untuk menampilkan gambar di Jupyter
def imshow(title, img, cmap=None):
    plt.figure(figsize=(6, 4))
    if len(img.shape) == 3:
        # BGR to RGB untuk Matplotlib
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(img, cmap='gray' if cmap is None else cmap)
    plt.title(title)
    plt.axis('off')
    plt.show()
"""

text_tuning = """\
## 1. Fase Prototipe: Tuning Nilai HSV Interaktif
Bagian ini digunakan untuk bereksperimen dengan gambar sampel agar mendapatkan nilai lower dan upper HSV yang pas untuk setiap warna helm (contoh: Kuning).
"""

code_tuning = """\
# Mengambil satu contoh gambar untuk tuning (ganti nama file sesuai yang ada di dataset Anda)
# NOTE: Gunakan garis miring biasa (/) untuk memisahkan folder
sample_image_path = "dataset/train/images/0_27797870_jpg.rf.JXUAgoNfPFI0KASkFRKb.jpg" # Ganti dengan salah satu file yang ada

if not os.path.exists(sample_image_path):
    print("Gambar tidak ditemukan. Silakan ganti 'sample_image_path' ke file yang valid!")
else:
    img_bgr = cv2.imread(sample_image_path)
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    @interact
    def tune_hsv(
        h_min=widgets.IntSlider(min=0, max=179, step=1, value=15),
        s_min=widgets.IntSlider(min=0, max=255, step=1, value=100),
        v_min=widgets.IntSlider(min=0, max=255, step=1, value=100),
        h_max=widgets.IntSlider(min=0, max=179, step=1, value=35),
        s_max=widgets.IntSlider(min=0, max=255, step=1, value=255),
        v_max=widgets.IntSlider(min=0, max=255, step=1, value=255)
    ):
        lower_bound = np.array([h_min, s_min, v_min])
        upper_bound = np.array([h_max, s_max, v_max])
        
        # Thresholding (Masking)
        mask = cv2.inRange(img_hsv, lower_bound, upper_bound)
        result = cv2.bitwise_and(img_bgr, img_bgr, mask=mask)
        
        plt.figure(figsize=(15,5))
        plt.subplot(1, 3, 1)
        plt.imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        plt.title('Citra Asli')
        plt.axis('off')
        
        plt.subplot(1, 3, 2)
        plt.imshow(mask, cmap='gray')
        plt.title('Masking HSV')
        plt.axis('off')
        
        plt.subplot(1, 3, 3)
        plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        plt.title('Hasil Masking')
        plt.axis('off')
        
        plt.show()
"""

text_multi = """\
## 2. Klasifikasi Multi-Warna Otomatis
Pada bagian ini, kita mengecek **beberapa warna sekaligus** pada sebuah gambar. Sistem akan melakukan masking untuk setiap warna dan menghitung jumlah piksel tepi (Sobel) mana yang paling mendominasi.
"""

code_multi = """\
# Kamus (Dictionary) rentang HSV untuk masing-masing warna Helm
color_ranges = {
    'Kuning': {
        'lower': np.array([15, 100, 100]),
        'upper': np.array([35, 255, 255])
    },
    'Merah': {
        # Merah memiliki dua rentang di HSV (di awal dan di akhir spektrum H)
        # Untuk simplifikasi, kita gunakan range utama (0-10)
        'lower': np.array([0, 100, 100]),
        'upper': np.array([10, 255, 255])
    },
    'Biru': {
        'lower': np.array([90, 100, 100]),
        'upper': np.array([130, 255, 255])
    },
    'Hijau': {
        'lower': np.array([40, 100, 100]),
        'upper': np.array([85, 255, 255])
    },
    'Putih': {
        # Putih memiliki Saturation rendah dan Value sangat tinggi
        'lower': np.array([0, 0, 200]),
        'upper': np.array([179, 50, 255])
    }
}

def detect_dominant_helmet_color(image_path, show_plot=False):
    img = cv2.imread(image_path)
    if img is None:
        return "Image not found"
        
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    max_edges = 0
    detected_color = "Tidak Terdeteksi (Unknown)"
    threshold_value = 500 # Minimal piksel Sobel untuk dianggap valid
    
    # Simpan hasil masking untuk diplot jika diperlukan
    best_mask = None
    best_sobel = None
    
    for color_name, bounds in color_ranges.items():
        # Masking
        mask = cv2.inRange(hsv, bounds['lower'], bounds['upper'])
        
        # Deteksi Tepi (Sobel)
        sobelx = cv2.Sobel(mask, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(mask, cv2.CV_64F, 0, 1, ksize=3)
        sobel_combined = np.uint8(np.absolute(cv2.magnitude(sobelx, sobely)))
        
        # Hitung jumlah piksel tepi
        edge_pixels = cv2.countNonZero(sobel_combined)
        
        if edge_pixels > max_edges:
            max_edges = edge_pixels
            best_mask = mask
            best_sobel = sobel_combined
            detected_color = color_name
            
    # Jika hasil maksimal ternyata di bawah standar kita
    if max_edges < threshold_value:
        detected_color = "Tidak Terdeteksi (Unknown)"
        
    if show_plot and detected_color != "Tidak Terdeteksi (Unknown)":
        plt.figure(figsize=(15, 4))
        plt.subplot(1, 3, 1)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title('Citra Asli')
        plt.axis('off')
        
        plt.subplot(1, 3, 2)
        plt.imshow(best_mask, cmap='gray')
        plt.title(f'Masking Warna Dominan ({detected_color})')
        plt.axis('off')
        
        plt.subplot(1, 3, 3)
        plt.imshow(best_sobel, cmap='gray')
        plt.title(f'Deteksi Tepi Sobel ({max_edges} px)')
        plt.axis('off')
        
        plt.show()
        
    return detected_color

# --- Uji Coba Multi-Warna ---
print(f"Hasil Klasifikasi: {detect_dominant_helmet_color(sample_image_path, show_plot=True)}")
"""

text_eval = """\
## 3. Evaluasi Dataset & Pelaporan Data (Laporan Bab 4)
Tahapan ini memproses seluruh gambar yang ada di dalam `dataset/train/images/`, menebak semua warna helm secara otomatis, lalu menyajikannya dalam bentuk Pandas DataFrame (Tabel) dan Bar Chart.
"""

code_eval = """\
import glob

# Path ke seluruh gambar dataset train
dataset_path = "dataset/train/images/*.jpg"
image_files = glob.glob(dataset_path)

print(f"Ditemukan {len(image_files)} gambar. Memulai proses evaluasi...")

results = []

# Kita batasi 200 gambar saja agar cepat, jika ingin semua hapus [:200]
for img_path in tqdm(image_files[:200], desc="Memproses Gambar"):
    color = detect_dominant_helmet_color(img_path, show_plot=False)
    
    # Ambil nama file asli (basename)
    filename = os.path.basename(img_path)
    
    results.append({
        "Nama File": filename,
        "Warna Terdeteksi": color
    })

# Konversi ke Pandas DataFrame (Tabel)
df_results = pd.DataFrame(results)

# 1. Menampilkan 10 baris pertama tabel
display(df_results.head(10))

# 2. Menghitung Rekapitulasi (Frequency)
summary_counts = df_results["Warna Terdeteksi"].value_counts()

# Menampilkan Tabel Rekapitulasi
summary_df = pd.DataFrame({
    'Warna': summary_counts.index,
    'Jumlah Terdeteksi': summary_counts.values
})
print("\\n--- TABEL REKAPITULASI ---")
display(summary_df)

# 3. Plot Grafik (Bar Chart & Pie Chart untuk Laporan)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Warna kustom menyesuaikan helm
colors = []
for idx in summary_counts.index:
    if idx == 'Kuning': colors.append('gold')
    elif idx == 'Merah': colors.append('red')
    elif idx == 'Biru': colors.append('blue')
    elif idx == 'Hijau': colors.append('green')
    elif idx == 'Putih': colors.append('whitesmoke')
    else: colors.append('gray')

# --- Bar Chart ---
bars = ax1.bar(summary_counts.index, summary_counts.values, color=colors, edgecolor='black')
ax1.set_title('Distribusi Deteksi Warna Helm', fontsize=14, fontweight='bold')
ax1.set_xlabel('Warna Helm', fontsize=12)
ax1.set_ylabel('Jumlah Gambar', fontsize=12)
ax1.grid(axis='y', linestyle='--', alpha=0.7)

# Tambahkan angka di atas bar
for bar in bars:
    yval = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2, yval + 0.5, int(yval), ha='center', va='bottom', fontweight='bold')

# --- Pie Chart (Diagram Lingkaran) ---
ax2.pie(summary_counts.values, labels=summary_counts.index, autopct='%1.1f%%', startangle=140, colors=colors, wedgeprops={'edgecolor': 'black'})
ax2.set_title('Persentase Deteksi', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig("Grafik_Laporan_Tubes.png", dpi=300) # Simpan grafik dengan resolusi tinggi
plt.show()

print("Grafik diagram berhasil disimpan sebagai 'Grafik_Laporan_Tubes.png'")

# 4. Export ke CSV
df_results.to_csv("hasil_evaluasi_helm.csv", index=False)
print("File 'hasil_evaluasi_helm.csv' telah berhasil dibuat di folder tugas besar Anda!")
"""

nb['cells'] = [
    nbf.v4.new_markdown_cell(text_intro),
    nbf.v4.new_code_cell(code_imports),
    nbf.v4.new_markdown_cell(text_tuning),
    nbf.v4.new_code_cell(code_tuning),
    nbf.v4.new_markdown_cell(text_multi),
    nbf.v4.new_code_cell(code_multi),
    nbf.v4.new_markdown_cell(text_eval),
    nbf.v4.new_code_cell(code_eval)
]

with open('Helmet_Color_Classification.ipynb', 'w') as f:
    nbf.write(nb, f)

print("Notebook berhasil diperbarui dengan fitur Multi-Warna dan Evaluasi Metrik!")
