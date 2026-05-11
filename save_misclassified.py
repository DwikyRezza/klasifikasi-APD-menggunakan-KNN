import cv2
import pandas as pd
import os
import shutil

def save_misclassified_images(csv_path, test_dir, output_dir):
    # 1. Buat folder output jika belum ada
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    
    # 2. Baca file CSV hasil evaluasi
    if not os.path.exists(csv_path):
        print(f"Error: File {csv_path} tidak ditemukan!")
        return
        
    df = pd.read_csv(csv_path)
    
    # 3. Filter data yang "SALAH"
    df_salah = df[df['Status'] == 'SALAH']
    
    if len(df_salah) == 0:
        print("Hebat! Tidak ada prediksi yang salah.")
        return
        
    print(f"Menemukan {len(df_salah)} prediksi salah. Sedang menyimpan ke folder '{output_dir}'...")
    
    count = 0
    for _, row in df_salah.iterrows():
        file_name = row['File']
        aktual = row['Aktual'].lower() # helm atau kacamata
        prediksi = row['Prediksi']
        
        # Cari path asli gambar di folder dataset/test
        img_path = os.path.join(test_dir, aktual, file_name)
        
        if not os.path.exists(img_path):
            # Coba cari tanpa subfolder jika tidak ketemu (fallback)
            img_path = os.path.join(test_dir, file_name)
            
        if os.path.exists(img_path):
            img = cv2.imread(img_path)
            if img is not None:
                # Tambahkan overlay teks (Aktual vs Prediksi)
                h, w, _ = img.shape
                
                # Background hitam kecil untuk teks agar terbaca
                cv2.rectangle(img, (0, 0), (w, 60), (0, 0, 0), -1)
                
                text_aktual = f"Aktual: {row['Aktual']}"
                text_pred = f"Pred: {prediksi}"
                
                cv2.putText(img, text_aktual, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(img, text_pred, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2) # Merah untuk salah
                
                # Simpan ke folder output
                save_path = os.path.join(output_dir, f"SALAH_{file_name}")
                cv2.imwrite(save_path, img)
                count += 1
        else:
            print(f"Peringatan: File {file_name} tidak ditemukan di {img_path}")

    print(f"Selesai! {count} gambar berhasil disimpan di folder '{output_dir}'.")

if __name__ == "__main__":
    CSV_FILE = "hasil_evaluasi_rf.csv"
    TEST_DIR = "dataset/test"
    OUTPUT_DIR = "hasil_salah"
    
    save_misclassified_images(CSV_FILE, TEST_DIR, OUTPUT_DIR)
