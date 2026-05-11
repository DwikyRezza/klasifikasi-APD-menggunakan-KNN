import nbformat as nbf
import os

nb = nbf.v4.new_notebook()

text_intro = """\
# Klasifikasi APD (Helm vs Kacamata) dengan Machine Learning (Random Forest)

Proyek ini mengklasifikasikan objek APD (Helm vs Kacamata) menggunakan pendekatan **Machine Learning**.

**Pipeline Lengkap:**
1. **Ekstraksi Fitur Geometri** dari citra (5 Fitur: Aspect Ratio, Circularity, Solidity, Extent, Hull AR) menggunakan Morphological Closing.
2. **Training Model Random Forest** pada data `train`.
3. **Visualisasi Decision Boundary**
4. **Evaluasi pada data `test`** menggunakan Confusion Matrix, Accuracy, Precision, Recall, F1-Score.
"""

code_imports = """\
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import glob
import pandas as pd
from tqdm.notebook import tqdm
import time
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (confusion_matrix, ConfusionMatrixDisplay,
                             accuracy_score, precision_score,
                             recall_score, f1_score, classification_report)
from sklearn.model_selection import train_test_split, learning_curve, StratifiedKFold, GridSearchCV, cross_val_score
from IPython.display import display

print(" Semua library berhasil diimport.")
print(f"   scikit-learn version: {__import__('sklearn').__version__}")
"""

text_extract = """\
## Tahap 1a: Ekstraksi Fitur dari Data Train

Sistem akan memproses semua foto di folder `dataset/train`.
Untuk setiap foto:
1. Konversi ke Grayscale & Blur.
2. Thresholding (Otsu).
3. **Morphological Closing**: Menyatukan kontur yang terputus agar bentuk kacamata/helm lebih utuh.
4. Pencarian kontur terbesar.
5. Ekstraksi 5 Fitur Geometri:
   - **Aspect Ratio**: Perbandingan lebar dan tinggi bounding box.
   - **Circularity**: Tingkat kebulatan objek (mendekati 1 jika bulat seperti helm).
   - **Solidity**: Kepadatan objek (Area kontur / Area Convex Hull). Helm padat, kacamata berongga.
   - **Extent**: Rasio area kontur terhadap area bounding box.
   - **Hull AR**: Aspect Ratio dari Convex Hull.

Hasil ekstraksi disimpan ke file `features_train.csv`.
"""

code_feature_extraction = """\
def extract_features(image_path):
    \"\"\"
    Ekstrak 5 fitur geometri dari sebuah gambar menggunakan Morphological Closing.
    Return: (aspect_ratio, circularity, solidity, extent, hull_ar) atau None jika gagal.
    \"\"\"
    img = cv2.imread(image_path)
    if img is None:
        return None

    # 1. Binarization (Otsu)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(img_gray, (7, 7), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    # 2. Morphological Closing: menyatukan kontur yang terputus
    kernel = np.ones((7, 7), np.uint8)
    binary_closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=3)

    # 3. Cari Kontur terbesar
    contours, _ = cv2.findContours(binary_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    best_contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(best_contour)
    if area < 500:
        return None

    # 4. Hitung 5 Fitur Geometri
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

    # FILTER SAMPAH: Helm/Kacamata tidak mungkin sangat lebar (AR > 2.0)
    # Jika AR > 2.0, biasanya itu adalah garis background atau gangguan.
    if aspect_ratio > 2.0 or aspect_ratio < 0.4:
        return None

    return aspect_ratio, circularity, solidity, extent, hull_ar

# --- EKSTRAKSI FITUR DARI FOLDER TRAIN ---
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

print(f"\\n Ekstraksi selesai! Total sampel valid: {len(df_train)}")
print(df_train.groupby("label").size().to_string())
display(df_train.head(10))
"""

text_eda = """\
## Tahap 1b: Eksplorasi Data (EDA)

Mari kita lihat apakah fitur-fitur ini mampu membedakan Helm dan Kacamata secara visual.
"""

code_eda = """\
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Distribusi Fitur Geometri per Kelas (Data Train)', fontsize=15, fontweight='bold')

colors = {"helm": "#2196F3", "kacamata": "#F44336"}
markers = {"helm": "o", "kacamata": "^"}

# Plot 1: AR vs Circularity
for label, group in df_train.groupby("label"):
    axes[0,0].scatter(group["aspect_ratio"], group["circularity"],
                    label=label.capitalize(), alpha=0.5, s=40,
                    color=colors[label], marker=markers[label])
axes[0,0].set_xlabel("Aspect Ratio")
axes[0,0].set_ylabel("Circularity")
axes[0,0].set_title("Aspect Ratio vs Circularity")
axes[0,0].legend()
axes[0,0].grid(True, alpha=0.3)

# Plot 2: Solidity vs Extent
for label, group in df_train.groupby("label"):
    axes[0,1].scatter(group["solidity"], group["extent"],
                    label=label.capitalize(), alpha=0.5, s=40,
                    color=colors[label], marker=markers[label])
axes[0,1].set_xlabel("Solidity (Kepadatan)")
axes[0,1].set_ylabel("Extent")
axes[0,1].set_title("Solidity vs Extent")
axes[0,1].legend()
axes[0,1].grid(True, alpha=0.3)

# Plot 3: Box plot Circularity
df_train.boxplot(column="circularity", by="label", ax=axes[1,0],
                 medianprops=dict(color="red", linewidth=2))
axes[1,0].set_title("Box Plot: Circularity")
axes[1,0].set_xlabel("Kelas")

# Plot 4: Box plot Solidity
df_train.boxplot(column="solidity", by="label", ax=axes[1,1],
                 medianprops=dict(color="red", linewidth=2))
axes[1,1].set_title("Box Plot: Solidity")
axes[1,1].set_xlabel("Kelas")

plt.suptitle("")
plt.tight_layout()
plt.show()
print(" Fitur yang baik akan menunjukkan pemisahan yang jelas antara Helm dan Kacamata.")
"""

text_train = """\
## Tahap 2a: Pelatihan Model Random Forest (Optimasi Lanjutan)

Data fitur tadi akan kita bersihkan dari outlier, lalu di-"ajarkan" ke model Machine Learning.
Langkah tingkat lanjut yang dilakukan:
1. **Outlier Removal (IQR)**: Membuang anomali pada fitur geometri.
2. **Hyperparameter Tuning (GridSearchCV)**: Mencari kombinasi parameter terbaik untuk Random Forest.
3. **5-Fold Cross Validation**: Menguji keandalan model secara silang.
4. **Feature Importance**: Menganalisis fitur mana yang paling penting.
"""

code_training = """\
# PENTING: Shuffle dulu agar Helm & Kacamata tercampur rata
df_train_shuffled = df_train.sample(frac=1, random_state=42).reset_index(drop=True)

# 1. CLEANING OUTLIER DENGAN IQR (Interquartile Range)
# Membuang outlier ekstrim untuk membersihkan data dari kesalahan ekstraksi background
Q1 = df_train_shuffled[['aspect_ratio', 'circularity', 'solidity']].quantile(0.05)
Q3 = df_train_shuffled[['aspect_ratio', 'circularity', 'solidity']].quantile(0.95)
IQR = Q3 - Q1
df_clean = df_train_shuffled[~((df_train_shuffled[['aspect_ratio', 'circularity', 'solidity']] < (Q1 - 1.5 * IQR)) | (df_train_shuffled[['aspect_ratio', 'circularity', 'solidity']] > (Q3 + 1.5 * IQR))).any(axis=1)]

print(f"Data awal: {len(df_train_shuffled)}, Setelah hapus outlier ekstrim: {len(df_clean)}")
print(f"Distribusi Kelas: {df_clean.groupby('label').size().to_dict()}")

X = df_clean[["aspect_ratio", "circularity", "solidity", "extent", "hull_ar"]].values
y = df_clean["label_id"].values
feature_names = ["Aspect Ratio", "Circularity", "Solidity", "Extent", "Hull AR"]

# Split untuk validasi internal (80/20 dari data train)
X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 2. HYPERPARAMETER TUNING DENGAN GRIDSEARCHCV
pipeline_template = Pipeline([
    ("scaler", StandardScaler()),
    ("rf", RandomForestClassifier(class_weight="balanced", random_state=42))
])

param_grid = {
    'rf__n_estimators': [100, 200],
    'rf__max_depth': [None, 10, 20],
    'rf__min_samples_split': [2, 5]
}

print("\\n Mencari hyperparameter terbaik (GridSearchCV)...")
grid_search = GridSearchCV(pipeline_template, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_tr, y_tr)

rf_pipeline = grid_search.best_estimator_
print(f" Parameter terbaik: {grid_search.best_params_}")

val_acc = rf_pipeline.score(X_val, y_val)
print(f" Akurasi pada data validasi internal (20%): {val_acc*100:.2f}%")

# 3. K-FOLD CROSS VALIDATION
cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(rf_pipeline, X, y, cv=cv_strategy, scoring='accuracy', n_jobs=-1)
print(f"\\n 5-Fold Cross Validation Scores: {[round(s*100, 2) for s in cv_scores]}")
print(f" Rata-rata CV Accuracy: {cv_scores.mean()*100:.2f}% (+/- {cv_scores.std()*100:.2f}%)")

# Simpan model
joblib.dump(rf_pipeline, "rf_apd_model.pkl")
print("\\n Model disimpan ke: rf_apd_model.pkl")

# --- GRAFIK 1: Feature Importance ---
importances = rf_pipeline.named_steps['rf'].feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 5))
plt.title("Feature Importance (Tingkat Kepentingan Fitur)", fontsize=14, fontweight='bold')
plt.bar(range(X.shape[1]), importances[indices], align="center", color="#FF9800", edgecolor='black')
plt.xticks(range(X.shape[1]), [feature_names[i] for i in indices], rotation=15, fontsize=12)
plt.ylabel("Skor Kepentingan", fontsize=12)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()

# --- GRAFIK 2: Learning Curve ---
train_sizes, train_scores, val_scores = learning_curve(
    rf_pipeline, X, y,
    train_sizes=np.linspace(0.3, 1.0, 8),
    cv=cv_strategy, scoring="accuracy", n_jobs=-1
)

train_mean = train_scores.mean(axis=1) * 100
train_std  = train_scores.std(axis=1) * 100
val_mean   = val_scores.mean(axis=1) * 100
val_std    = val_scores.std(axis=1) * 100

plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_mean, "o-", color="#2196F3", label="Akurasi Training")
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.15, color="#2196F3")
plt.plot(train_sizes, val_mean, "s--", color="#F44336", label="Akurasi Validasi (CV-5)")
plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.15, color="#F44336")

plt.xlabel("Jumlah Data Training", fontsize=13)
plt.ylabel("Akurasi (%)", fontsize=13)
plt.title("Learning Curve - Akurasi vs Jumlah Data Training", fontsize=14, fontweight="bold")
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.ylim([40, 105])
plt.tight_layout()
plt.show()
print(" Grafik Learning Curve stabil, dan Feature Importance menunjukkan fitur yang paling dominan.")
"""

text_boundary = """\
## Tahap 2b: Visualisasi Decision Boundary

Visualisasi ini menunjukkan **"area yang dipelajari" oleh model Random Forest**.
Karena kita punya 5 fitur, kita tidak bisa memvisualisasikan 5D dalam grafik 2D.
Solusinya: Kita buat irisan (slice) pada Aspect Ratio vs Circularity,
sementara fitur lainnya (Solidity, Extent, Hull AR) kita set ke nilai rata-rata (mean).
"""

code_boundary = """\
x_min, x_max = X[:, 0].min() - 0.3, X[:, 0].max() + 0.3
y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 400),
                     np.linspace(y_min, y_max, 400))

mean_solidity = X[:, 2].mean()
mean_extent = X[:, 3].mean()
mean_hull_ar = X[:, 4].mean()

grid_2d = np.c_[xx.ravel(), yy.ravel()]
n_samples = grid_2d.shape[0]
grid_5d = np.c_[
    grid_2d, 
    np.full(n_samples, mean_solidity),
    np.full(n_samples, mean_extent),
    np.full(n_samples, mean_hull_ar)
]

Z = rf_pipeline.predict(grid_5d).reshape(xx.shape)

plt.figure(figsize=(11, 7))
plt.contourf(xx, yy, Z, alpha=0.25, cmap=plt.cm.RdBu)
plt.contour(xx, yy, Z, colors="black", linewidths=1.5, linestyles="--")

colors_map = {0: "#1565C0", 1: "#C62828"}
labels_map = {0: "Helm (Train)", 1: "Kacamata (Train)"}
markers_map = {0: "o", 1: "^"}

for cls in [0, 1]:
    mask = y == cls
    plt.scatter(X[mask, 0], X[mask, 1],
                c=colors_map[cls], marker=markers_map[cls],
                label=labels_map[cls], edgecolors="white", s=70, alpha=0.85)

patch_helm = mpatches.Patch(color="#90CAF9", label="Zona Helm (Model)")
patch_kaca = mpatches.Patch(color="#EF9A9A", label="Zona Kacamata (Model)")
handles, lgs = plt.gca().get_legend_handles_labels()
plt.legend(handles=handles + [patch_helm, patch_kaca], fontsize=10, loc="upper right")

plt.xlabel("Aspect Ratio (Lebar/Tinggi)", fontsize=13)
plt.ylabel("Circularity (Kebulatan)", fontsize=13)
plt.title("Decision Boundary Random Forest (Slice 2D)", fontsize=14, fontweight="bold")
plt.grid(True, alpha=0.2)
plt.tight_layout()
plt.show()
"""

text_classify = """\
## Tahap 3: Fungsi Klasifikasi Menggunakan Model Random Forest

Menggantikan logika `if aspect_ratio > 1.5` dengan `model.predict()`.
200 pohon keputusan bekerja bersama (voting) untuk menentukan label akhir berdasarkan 5 fitur.
"""

code_classify = """\
def classify_with_model(image_path, pipeline, show_plot=False):
    \"\"\"
    Ekstrak 5 fitur dari gambar lalu klasifikasikan menggunakan model Random Forest.
    Menggantikan logika rule-based.
    \"\"\"
    start_time = time.time()
    img = cv2.imread(image_path)
    if img is None:
        return "Error", 0, 0, 0, 0, 0, 0

    # Image Processing (identik dengan extract_features saat training)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(img_gray, (7, 7), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    kernel = np.ones((7, 7), np.uint8)
    binary_closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=3)

    contours, _ = cv2.findContours(binary_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    predicted_label = "Tidak Dikenali"
    aspect_ratio, circularity, solidity, extent, hull_ar = 0, 0, 0, 0, 0
    area = 0
    best_contour, x, y, w, h = None, 0, 0, 0, 0

    if contours:
        best_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(best_contour)

        if area > 500:
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

            # Kirim 5 fitur ke model
            features = np.array([[aspect_ratio, circularity, solidity, extent, hull_ar]])
            pred_id = pipeline.predict(features)[0]
            predicted_label = "Helm" if pred_id == 0 else "Kacamata"

    end_time = time.time()

    if show_plot:
        img_result = img.copy()
        if best_contour is not None and area > 500:
            color = (0, 200, 0) if predicted_label == "Helm" else (0, 0, 220)
            cv2.rectangle(img_result, (x, y), (x+w, y+h), color, 3)
            cv2.putText(img_result, f"{predicted_label} | AR:{aspect_ratio:.2f} Sol:{solidity:.2f}",
                        (x, max(y-10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)); axes[0].set_title("1. Citra Asli"); axes[0].axis("off")
        axes[1].imshow(binary, cmap="gray"); axes[1].set_title("2. Biner (Otsu)"); axes[1].axis("off")
        axes[2].imshow(binary_closed, cmap="gray"); axes[2].set_title("3. Morph. Closing"); axes[2].axis("off")
        axes[3].imshow(cv2.cvtColor(img_result, cv2.COLOR_BGR2RGB)); axes[3].set_title(f"4. Hasil RF: {predicted_label}"); axes[3].axis("off")
        plt.tight_layout(); plt.show()

    return predicted_label, aspect_ratio, circularity, solidity, extent, hull_ar, end_time - start_time

# --- DEMO pada 1 sampel helm dan 1 sampel kacamata ---
print("=== DEMO KLASIFIKASI MENGGUNAKAN MODEL RANDOM FOREST ===")
helm_samples = glob.glob("dataset/test/helm/*.*")
if helm_samples:
    print("\\n[HELM]")
    classify_with_model(helm_samples[0], rf_pipeline, show_plot=True)

kacamata_samples = glob.glob("dataset/test/kacamata/*.*")
if kacamata_samples:
    print("\\n[KACAMATA]")
    classify_with_model(kacamata_samples[0], rf_pipeline, show_plot=True)
"""

text_eval = """\
## Tahap 4: Evaluasi pada Data Test (Unseen Data)

Model yang sudah dilatih akan diuji pada sekumpulan foto di folder `dataset/test`.
Hasilnya akan dievaluasi menggunakan **Confusion Matrix** dan **Classification Report**.
"""

code_eval = """\
test_dir = "dataset/test"
test_files = glob.glob(os.path.join(test_dir, "*", "*.*"))

y_true = []
y_pred = []
results = []

print(f" Memproses {len(test_files)} gambar data test...")

for img_path in tqdm(test_files, desc="Evaluasi Test Data"):
    parent = os.path.basename(os.path.dirname(img_path))
    actual = "Helm" if parent.lower() == "helm" else "Kacamata"
    y_true.append(actual)

    pred, ar, circ, sol, ext, har, p_time = classify_with_model(img_path, rf_pipeline, show_plot=False)
    y_pred.append(pred)
    results.append({
        "File": os.path.basename(img_path),
        "Aktual": actual,
        "Prediksi": pred,
        "Status": "BENAR" if actual == pred else "SALAH",
        "Aspect Ratio": round(ar, 2),
        "Circularity": round(circ, 2),
        "Solidity": round(sol, 2)
    })

df_results = pd.DataFrame(results)
df_results.to_csv("hasil_evaluasi_rf.csv", index=False)
print(f"\\n Hasil disimpan ke hasil_evaluasi_rf.csv")
display(df_results.head(10))

# --- CONFUSION MATRIX ---
labels = ["Helm", "Kacamata"]
cm = confusion_matrix(y_true, y_pred, labels=labels)

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot(ax=axes[0], cmap="Blues", colorbar=False)
axes[0].set_title("Confusion Matrix\\nModel Random Forest (Data Test)", fontsize=13, fontweight="bold")

# --- METRIK EVALUASI ---
acc  = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred, pos_label="Helm", zero_division=0)
rec  = recall_score(y_true, y_pred, pos_label="Helm", zero_division=0)
f1   = f1_score(y_true, y_pred, pos_label="Helm", zero_division=0)

metrics = ["Accuracy", "Precision (Helm)", "Recall (Helm)", "F1-Score (Helm)"]
values = [acc * 100, prec * 100, rec * 100, f1 * 100]
bar_colors = ["#4CAF50", "#2196F3", "#FF9800", "#9C27B0"]

bars = axes[1].bar(metrics, values, color=bar_colors, edgecolor="white", linewidth=1.5)
axes[1].set_ylim([0, 110])
axes[1].set_ylabel("Nilai (%)", fontsize=12)
axes[1].set_title("Metrik Evaluasi Model Random Forest", fontsize=13, fontweight="bold")
axes[1].grid(axis="y", alpha=0.3)
for bar, val in zip(bars, values):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
                 f"{val:.1f}%", ha='center', va='bottom', fontweight='bold', fontsize=11)

plt.tight_layout()
plt.show()

print("\\n--- CLASSIFICATION REPORT ---")
print(classification_report(y_true, y_pred, labels=labels, zero_division=0))

print(f"\\n{'='*40}")
print(f"  RINGKASAN PERFORMA MODEL RANDOM FOREST")
print(f"{'='*40}")
print(f"  Accuracy  : {acc*100:.2f}%")
print(f"  Precision : {prec*100:.2f}%")
print(f"  Recall    : {rec*100:.2f}%")
print(f"  F1-Score  : {f1*100:.2f}%")
print(f"{'='*40}")
"""

nb.cells = [
    nbf.v4.new_markdown_cell(text_intro),
    nbf.v4.new_code_cell(code_imports),
    nbf.v4.new_markdown_cell(text_extract),
    nbf.v4.new_code_cell(code_feature_extraction),
    nbf.v4.new_markdown_cell(text_eda),
    nbf.v4.new_code_cell(code_eda),
    nbf.v4.new_markdown_cell(text_train),
    nbf.v4.new_code_cell(code_training),
    nbf.v4.new_markdown_cell(text_boundary),
    nbf.v4.new_code_cell(code_boundary),
    nbf.v4.new_markdown_cell(text_classify),
    nbf.v4.new_code_cell(code_classify),
    nbf.v4.new_markdown_cell(text_eval),
    nbf.v4.new_code_cell(code_eval)
]

with open("Helmet_Color_Classification.ipynb", "w", encoding='utf-8') as f:
    nbf.write(nb, f)

print(" Notebook berhasil digenerate dengan pipeline ML Random Forest (5 Fitur) lengkap!")
print("   Jalankan: jupyter notebook Helmet_Color_Classification.ipynb")
