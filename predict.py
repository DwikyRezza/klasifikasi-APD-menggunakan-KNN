import cv2
import numpy as np
import joblib
import sys
import os
import matplotlib.pyplot as plt

def extract_features_single(img):
    """Fungsi ekstraksi fitur yang persis sama dengan saat training."""
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(img_gray, (7, 7), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    kernel = np.ones((7, 7), np.uint8)
    binary_closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=3)

    contours, _ = cv2.findContours(binary_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None, None

    best_contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(best_contour)
    if area < 500:
        return None, None, None

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

    features = [aspect_ratio, circularity, solidity, extent, hull_ar]
    return features, best_contour, (x, y, w, h)

def predict_image(image_path, model_path="rf_apd_model.pkl"):
    if not os.path.exists(image_path):
        print(f"Error: Gambar {image_path} tidak ditemukan.")
        return
        
    if not os.path.exists(model_path):
        print(f"Error: Model {model_path} tidak ditemukan.")
        return

    # Load Model
    model = joblib.load(model_path)
    
    # Read Image
    img = cv2.imread(image_path)
    if img is None:
        print("Error membaca gambar.")
        return
        
    img_disp = img.copy()

    # Extract Features
    features, contour, bbox = extract_features_single(img)
    
    if features is None:
        print("Tidak ada objek yang terdeteksi dengan jelas pada gambar.")
        label = "Tidak Terdeteksi"
    else:
        # Predict
        features_array = np.array([features])
        pred_id = model.predict(features_array)[0]
        label = "Helm" if pred_id == 0 else "Kacamata"
        
        # Draw Bounding Box and Label
        x, y, w, h = bbox
        color = (0, 255, 0) if label == "Helm" else (0, 0, 255)
        cv2.rectangle(img_disp, (x, y), (x+w, y+h), color, 3)
        cv2.putText(img_disp, f"{label}", (x, max(y-10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        
        print(f"Hasil Prediksi: {label}")
        print(f"Fitur: AR={features[0]:.2f}, Circ={features[1]:.2f}, Sol={features[2]:.2f}")

    # Tampilkan Gambar menggunakan Matplotlib
    plt.figure(figsize=(8, 6))
    plt.imshow(cv2.cvtColor(img_disp, cv2.COLOR_BGR2RGB))
    plt.title(f"Hasil Deteksi: {label}", fontsize=16, fontweight='bold')
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    print("=== SCRIPT INFERENCE REAL-TIME ===")
    if len(sys.argv) > 1:
        img_path = sys.argv[1]
    else:
        # Coba ambil satu sampel jika tidak ada argumen
        import glob
        samples = glob.glob("dataset/test/*/*.*")
        if samples:
            img_path = samples[0]
            print(f"Menggunakan sampel otomatis: {img_path}")
        else:
            print("Usage: python predict.py <path_ke_gambar>")
            sys.exit(1)
            
    # Gunakan model tuned jika ada, jika tidak gunakan model biasa
    model_to_use = "rf_apd_model_tuned.pkl" if os.path.exists("rf_apd_model_tuned.pkl") else "rf_apd_model.pkl"
    print(f"Menggunakan model: {model_to_use}")
    predict_image(img_path, model_path=model_to_use)
