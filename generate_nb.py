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
sample_image_path = "train/images/000001_jpg.rf.f81ab748f431180301ee5069e82ab67c.jpg" # Ganti dengan salah satu file yang ada

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

text_main = """\
## 2. Implementasi Alur Utama (Klasifikasi Penuh)
Menggunakan nilai HSV yang sudah di-tuning di atas untuk menjalankan alur:
**Citra Asli -> Konversi HSV -> Masking -> Deteksi Tepi (Sobel) -> Klasifikasi**
"""

code_main = """\
def process_and_classify(image_path):
    # 1. Citra Asli
    img = cv2.imread(image_path)
    if img is None:
        return "Image not found"
    
    # 2. Konversi HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Range Warna (Contoh: Warna Kuning, sesuaikan dengan hasil tuning Anda)
    # Anda dapat menambahkan dictionary/kondisi untuk mengecek warna lain (Merah, Biru, Putih)
    lower_yellow = np.array([15, 100, 100])
    upper_yellow = np.array([35, 255, 255])
    
    # 3. Thresholding Warna (Masking)
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    
    # 4. Deteksi Tepi (Sobel)
    # Kita aplikasikan Sobel pada hasil Masking untuk mempertegas bentuk
    sobelx = cv2.Sobel(mask, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(mask, cv2.CV_64F, 0, 1, ksize=3)
    sobel_combined = cv2.magnitude(sobelx, sobely)
    sobel_combined = np.uint8(np.absolute(sobel_combined))
    
    # Opsional: Jika Anda ingin menggunakan Prewitt
    # kernelx = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
    # kernely = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
    # prewittx = cv2.filter2D(mask, -1, kernelx)
    # prewitty = cv2.filter2D(mask, -1, kernely)
    # edges = prewittx + prewitty
    
    # 5. Klasifikasi
    # Menghitung seberapa banyak piksel edge (tepi) yang terdeteksi
    # Jika piksel putih melebihi threshold tertentu, maka helm terdeteksi warna tersebut
    edge_pixels = cv2.countNonZero(sobel_combined)
    threshold_value = 500 # Sesuaikan nilai ini dengan dataset Anda
    
    if edge_pixels > threshold_value:
        classification = "Helm Kuning Terdeteksi"
    else:
        classification = "Bukan Helm Kuning"
        
    # --- VISUALISASI HASIL LENGKAP ---
    plt.figure(figsize=(16, 4))
    
    plt.subplot(1, 4, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title('1. Citra Asli')
    plt.axis('off')
    
    plt.subplot(1, 4, 2)
    # Konversi HSV sebenarnya tidak bisa divisualisasikan RGB dengan baik, 
    # tapi kita tampilkan channel Hue sebagai representasi
    plt.imshow(hsv[:,:,0], cmap='hsv') 
    plt.title('2. Konversi HSV (Channel H)')
    plt.axis('off')
    
    plt.subplot(1, 4, 3)
    plt.imshow(mask, cmap='gray')
    plt.title('3. Masking Warna')
    plt.axis('off')
    
    plt.subplot(1, 4, 4)
    plt.imshow(sobel_combined, cmap='gray')
    plt.title(f'4. Sobel Edge\\nKlasifikasi: {classification}')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return classification

# Uji coba fungsi
# Ganti nama file ini dengan gambar lain di folder train/images/ untuk tes
test_file = sample_image_path 
print(f"Hasil Klasifikasi: {process_and_classify(test_file)}")
"""

nb['cells'] = [
    nbf.v4.new_markdown_cell(text_intro),
    nbf.v4.new_code_cell(code_imports),
    nbf.v4.new_markdown_cell(text_tuning),
    nbf.v4.new_code_cell(code_tuning),
    nbf.v4.new_markdown_cell(text_main),
    nbf.v4.new_code_cell(code_main)
]

with open('Helmet_Color_Classification.ipynb', 'w') as f:
    nbf.write(nb, f)

print("Notebook berhasil dibuat: Helmet_Color_Classification.ipynb")
