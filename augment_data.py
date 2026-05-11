import cv2
import os
import glob
from tqdm import tqdm

def augment_images(base_dir="dataset/train"):
    print("=== MULAI DATA AUGMENTATION ===")
    categories = ["helm", "kacamata"]
    
    total_augmented = 0
    
    for category in categories:
        folder_path = os.path.join(base_dir, category)
        if not os.path.exists(folder_path):
            print(f"Peringatan: Folder {folder_path} tidak ditemukan.")
            continue
            
        files = glob.glob(os.path.join(folder_path, "*.*"))
        # Filter agar tidak melakukan augmentasi pada gambar yang sudah diaugmentasi
        original_files = [f for f in files if "_aug_" not in f]
        
        print(f"\nMemproses {len(original_files)} gambar asli di kategori '{category}'...")
        
        for img_path in tqdm(original_files, desc=f"Augmenting {category}"):
            img = cv2.imread(img_path)
            if img is None:
                continue
                
            base_name = os.path.basename(img_path)
            name, ext = os.path.splitext(base_name)
            
            # 1. Horizontal Flip
            flipped = cv2.flip(img, 1)
            flip_path = os.path.join(folder_path, f"{name}_aug_flip{ext}")
            cv2.imwrite(flip_path, flipped)
            total_augmented += 1
            
            # 2. Brightness Adjustment (agak gelap)
            matrix = np.ones(img.shape, dtype="uint8") * 30
            darker = cv2.subtract(img, matrix)
            dark_path = os.path.join(folder_path, f"{name}_aug_dark{ext}")
            cv2.imwrite(dark_path, darker)
            total_augmented += 1

            # 3. Brightness Adjustment (agak terang)
            lighter = cv2.add(img, matrix)
            light_path = os.path.join(folder_path, f"{name}_aug_light{ext}")
            cv2.imwrite(light_path, lighter)
            total_augmented += 1

    print(f"\nSelesai! {total_augmented} gambar augmentasi baru telah ditambahkan.")
    print("\nPENTING: Anda harus mengekstrak fitur ulang karena jumlah data bertambah!")
    print("Jalankan ulang sel ekstraksi fitur di jupyter notebook atau jalankan script ekstraksi.")

if __name__ == "__main__":
    import numpy as np
    augment_images()
