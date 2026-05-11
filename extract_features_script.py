import cv2
import numpy as np
import os
import glob
import pandas as pd
from tqdm import tqdm

def extract_features(image_path):
    img = cv2.imread(image_path)
    if img is None: return None
    
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(img_gray, (7, 7), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    kernel = np.ones((7, 7), np.uint8)
    binary_closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=3)

    contours, _ = cv2.findContours(binary_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return None

    best_contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(best_contour)
    if area < 500: return None

    perimeter = cv2.arcLength(best_contour, True)
    x, y, w, h = cv2.boundingRect(best_contour)
    hull = cv2.convexHull(best_contour)
    hull_area = cv2.contourArea(hull)
    hx, hy, hw, hh = cv2.boundingRect(hull)

    aspect_ratio = float(w) / h if h > 0 else 0
    circularity = (4 * np.pi * area / (perimeter ** 2)) if perimeter > 0 else 0
    solidity = area / hull_area if hull_area > 0 else 0
    extent = area / (w * h) if (w * h) > 0 else 0
    hull_ar = float(hw) / hh if hh > 0 else 0

    if aspect_ratio < 0.4:
        return None

    return aspect_ratio, circularity, solidity, extent, hull_ar

print("=== EKSTRAKSI FITUR ===")
train_dir = "dataset/train"
categories = {"helm": 0, "kacamata": 1}
records = []

for label_name, label_id in categories.items():
    folder = os.path.join(train_dir, label_name)
    files = glob.glob(os.path.join(folder, "*.*"))
    print(f"Memproses '{label_name}': {len(files)} gambar ditemukan...")

    for img_path in tqdm(files, desc=f"Ekstraksi {label_name}"):
        result = extract_features(img_path)
        if result is not None:
            ar, circ, sol, ext, hull_ar = result
            records.append({
                "file": os.path.basename(img_path),
                "aspect_ratio": round(ar, 4),
                "circularity": round(circ, 4),
                "solidity": round(sol, 4),
                "extent": round(ext, 4),
                "hull_ar": round(hull_ar, 4),
                "label": label_name,
                "label_id": label_id
            })

df_train = pd.DataFrame(records)
df_train.to_csv("features_train.csv", index=False)
print(f"Ekstraksi selesai! Total sampel valid: {len(df_train)}")
