"""
Script untuk mengupgrade generate_nb.py dari versi lama (SVM + 2 fitur + Sobel)
ke versi baru (Random Forest + 5 fitur + Morphological Closing).
"""
import re

with open('generate_nb.py', 'r', encoding='utf-8') as f:
    content = f.read()

print(f"File asli: {len(content.splitlines())} baris")

# ===================================================
# PATCH 1: Ganti import SVC -> RandomForestClassifier
# ===================================================
content = content.replace(
    'from sklearn.svm import SVC',
    'from sklearn.ensemble import RandomForestClassifier'
)
print("PATCH 1 (import) - OK")

# ===================================================
# PATCH 2: Upgrade extract_features (2 fitur -> 5 fitur + Morph Closing)
# ===================================================
old_extract = '''    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(img_gray, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    # Deteksi Tepi Sobel
    sobelx = cv2.Sobel(binary, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(binary, cv2.CV_64F, 0, 1, ksize=3)
    sobel_combined = np.uint8(np.absolute(cv2.magnitude(sobelx, sobely)))

    contours, _ = cv2.findContours(sobel_combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    best_contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(best_contour)
    if area < 500:
        return None

    perimeter = cv2.arcLength(best_contour, True)
    x, y, w, h = cv2.boundingRect(best_contour)

    aspect_ratio = float(w) / h if h > 0 else 0
    circularity = (4 * np.pi * area / (perimeter ** 2)) if perimeter > 0 else 0

    return aspect_ratio, circularity'''

new_extract = '''    # 1. Binarization (Otsu)
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

    return aspect_ratio, circularity, solidity, extent, hull_ar'''

if old_extract in content:
    content = content.replace(old_extract, new_extract)
    print("PATCH 2 (extract_features) - OK")
else:
    print("PATCH 2 - GAGAL, cari manual...")

# ===================================================
# PATCH 3: Update records.append agar simpan 5 fitur
# ===================================================
old_records = '''        if result is not None:
            ar, circ = result
            records.append({
                "file": os.path.basename(img_path),
                "aspect_ratio": round(ar, 4),
                "circularity": round(circ, 4),
                "label": label_name,
                "label_id": label_id
            })'''

new_records = '''        if result is not None:
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
            })'''

if old_records in content:
    content = content.replace(old_records, new_records)
    print("PATCH 3 (records.append) - OK")
else:
    print("PATCH 3 - GAGAL")

# ===================================================
# PATCH 4: Ganti scatter plot 1x2 -> 2x2 dengan 4 grafik
# ===================================================
old_eda = '''fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle(\'Distribusi Fitur Geometri per Kelas (Data Train)\', fontsize=15, fontweight=\'bold\')

colors = {"helm": "#2196F3", "kacamata": "#F44336"}
markers = {"helm": "o", "kacamata": "^"}

for label, group in df_train.groupby("label"):
    axes[0].scatter(group["aspect_ratio"], group["circularity"],
                    label=label.capitalize(), alpha=0.6, s=60,
                    color=colors[label], marker=markers[label])

axes[0].set_xlabel("Aspect Ratio (Lebar/Tinggi)", fontsize=12)
axes[0].set_ylabel("Circularity (Kebulatan)", fontsize=12)
axes[0].set_title("Scatter Plot: Aspect Ratio vs Circularity")
axes[0].legend(fontsize=11)
axes[0].grid(True, alpha=0.3)

# Box plot distribusi aspect ratio
df_train.boxplot(column="aspect_ratio", by="label", ax=axes[1],
                 boxprops=dict(color="#333"),
                 medianprops=dict(color="red", linewidth=2))
axes[1].set_title("Box Plot: Aspect Ratio per Kelas")
axes[1].set_xlabel("Kelas")
axes[1].set_ylabel("Aspect Ratio")
plt.suptitle("")

plt.tight_layout()
plt.show()
print(" Jika kedua kelas terpisah di scatter plot, artinya fitur sangat baik untuk klasifikasi.")'''

new_eda = '''fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle(\'Distribusi Fitur Geometri per Kelas (Data Train)\', fontsize=15, fontweight=\'bold\')

colors = {"helm": "#2196F3", "kacamata": "#F44336"}
markers = {"helm": "o", "kacamata": "^"}

# Plot 1: AR vs Circularity
for label, group in df_train.groupby("label"):
    axes[0,0].scatter(group["aspect_ratio"], group["circularity"],
                    label=label.capitalize(), alpha=0.5, s=40,
                    color=colors[label], marker=markers[label])
axes[0,0].set_xlabel("Aspect Ratio"); axes[0,0].set_ylabel("Circularity")
axes[0,0].set_title("Aspect Ratio vs Circularity"); axes[0,0].legend(); axes[0,0].grid(True, alpha=0.3)

# Plot 2: Solidity vs Extent
for label, group in df_train.groupby("label"):
    axes[0,1].scatter(group["solidity"], group["extent"],
                    label=label.capitalize(), alpha=0.5, s=40,
                    color=colors[label], marker=markers[label])
axes[0,1].set_xlabel("Solidity (Kepadatan)"); axes[0,1].set_ylabel("Extent")
axes[0,1].set_title("Solidity vs Extent"); axes[0,1].legend(); axes[0,1].grid(True, alpha=0.3)

# Plot 3: Box plot Circularity
df_train.boxplot(column="circularity", by="label", ax=axes[1,0],
                 medianprops=dict(color="red", linewidth=2))
plt.sca(axes[1,0]); plt.title("Box Plot: Circularity")
axes[1,0].set_xlabel("Kelas")

# Plot 4: Box plot Solidity
df_train.boxplot(column="solidity", by="label", ax=axes[1,1],
                 medianprops=dict(color="red", linewidth=2))
plt.sca(axes[1,1]); plt.title("Box Plot: Solidity")
axes[1,1].set_xlabel("Kelas")

plt.suptitle("")
plt.tight_layout()
plt.show()
print(" Fitur yang baik akan menunjukkan pemisahan yang jelas antara warna biru dan merah.")'''

if old_eda in content:
    content = content.replace(old_eda, new_eda)
    print("PATCH 4 (EDA plot) - OK")
else:
    print("PATCH 4 - GAGAL")

# ===================================================
# PATCH 5: Ganti SVM pipeline -> Random Forest pipeline
# ===================================================
old_pipeline = '''# Pipeline: Scaler + SVM
svm_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, random_state=42))
])

print(" Melatih model SVM...")
svm_pipeline.fit(X_tr, y_tr)

val_acc = svm_pipeline.score(X_val, y_val)
print(f" Training selesai!")
print(f"   Akurasi pada data validasi internal: {val_acc*100:.2f}%")

# Simpan model
joblib.dump(svm_pipeline, "svm_apd_model.pkl")
print("   Model disimpan ke: svm_apd_model.pkl")'''

new_pipeline = '''# Pipeline: Random Forest dengan class_weight=\'balanced\'
# class_weight=\'balanced\' = model dihukum lebih berat jika salah nebak kelas minoritas (Kacamata)
rf_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("rf", RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ))
])

print(" Melatih model Random Forest (Balanced)...")
rf_pipeline.fit(X_tr, y_tr)

val_acc = rf_pipeline.score(X_val, y_val)
print(f" Training selesai!")
print(f"   Akurasi pada data validasi internal: {val_acc*100:.2f}%")

# Simpan model
joblib.dump(rf_pipeline, "rf_apd_model.pkl")
print("   Model disimpan ke: rf_apd_model.pkl")'''

if old_pipeline in content:
    content = content.replace(old_pipeline, new_pipeline)
    print("PATCH 5 (RF pipeline) - OK")
else:
    print("PATCH 5 - GAGAL")

# ===================================================
# PATCH 6: Update X untuk pakai 5 fitur
# ===================================================
old_x = 'X = df_train[["aspect_ratio", "circularity"]].values'
new_x = 'X = df_train_shuffled[["aspect_ratio", "circularity", "solidity", "extent", "hull_ar"]].values'
old_shuffle = 'X = df_train[["aspect_ratio", "circularity"]].values'

# Cek apakah sudah ada shuffle
if 'df_train_shuffled' in content:
    content = content.replace(
        'X = df_train_shuffled[["aspect_ratio", "circularity"]].values',
        'X = df_train_shuffled[["aspect_ratio", "circularity", "solidity", "extent", "hull_ar"]].values'
    )
    print("PATCH 6a (X 5-fitur dengan shuffle) - OK")
elif old_x in content:
    content = content.replace(old_x, new_x)
    print("PATCH 6b (X 5-fitur) - OK")
else:
    print("PATCH 6 - GAGAL")

# ===================================================
# PATCH 7: Tambah shuffle sebelum X (jika belum ada)
# ===================================================
if 'df_train_shuffled' not in content:
    old_x_line = 'X = df_train[["aspect_ratio", "circularity", "solidity", "extent", "hull_ar"]].values'
    new_x_with_shuffle = '''# PENTING: Shuffle dulu agar Helm & Kacamata tercampur rata
df_train_shuffled = df_train.sample(frac=1, random_state=42).reset_index(drop=True)
print(f"Data setelah di-shuffle: {df_train_shuffled.groupby(\'label\').size().to_dict()}")

X = df_train_shuffled[["aspect_ratio", "circularity", "solidity", "extent", "hull_ar"]].values'''
    if old_x_line in content:
        content = content.replace(old_x_line, new_x_with_shuffle)
        print("PATCH 7 (shuffle) - OK")

# ===================================================
# PATCH 8: Update learning_curve pakai rf_pipeline
# ===================================================
content = content.replace(
    'train_sizes, train_scores, val_scores = learning_curve(\n    svm_pipeline, X, y,',
    'train_sizes, train_scores, val_scores = learning_curve(\n    rf_pipeline, X, y,'
)
print("PATCH 8 (learning_curve) - OK")

# ===================================================
# PATCH 9: Decision Boundary - pakai grid 5D
# ===================================================
old_boundary = '''x_min, x_max = X[:, 0].min() - 0.3, X[:, 0].max() + 0.3
y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 400),
                     np.linspace(y_min, y_max, 400))

grid_input = np.c_[xx.ravel(), yy.ravel()]
Z = rf_pipeline.predict(grid_input).reshape(xx.shape)'''

new_boundary = '''x_min, x_max = X[:, 0].min() - 0.3, X[:, 0].max() + 0.3
y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 400),
                     np.linspace(y_min, y_max, 400))

# Model butuh 5 fitur. Kita visualisasikan AR vs Circularity,
# fitur lain di-set ke nilai rata-rata (mean slice).
mean_solidity = X[:, 2].mean()
mean_extent   = X[:, 3].mean()
mean_hull_ar  = X[:, 4].mean()

grid_2d = np.c_[xx.ravel(), yy.ravel()]
n_pts = grid_2d.shape[0]
grid_5d = np.c_[
    grid_2d,
    np.full(n_pts, mean_solidity),
    np.full(n_pts, mean_extent),
    np.full(n_pts, mean_hull_ar)
]
Z = rf_pipeline.predict(grid_5d).reshape(xx.shape)'''

if old_boundary in content:
    content = content.replace(old_boundary, new_boundary)
    print("PATCH 9 (decision boundary 5D) - OK")
else:
    print("PATCH 9 - GAGAL (mungkin sudah benar)")

# ===================================================
# PATCH 10: Update classify_with_model -> 5 fitur + Morph Closing
# ===================================================
old_classify_body = '''    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(img_gray, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    sobelx = cv2.Sobel(binary, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(binary, cv2.CV_64F, 0, 1, ksize=3)
    sobel_combined = np.uint8(np.absolute(cv2.magnitude(sobelx, sobely)))

    contours, _ = cv2.findContours(sobel_combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    predicted_label = "Tidak Dikenali"
    aspect_ratio, circularity = 0, 0
    area = 0
    best_contour, x, y, w, h = None, 0, 0, 0, 0

    if contours:
        best_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(best_contour)

        if area > 500:
            perimeter = cv2.arcLength(best_contour, True)
            x, y, w, h = cv2.boundingRect(best_contour)
            aspect_ratio = float(w) / h if h > 0 else 0
            circularity = (4 * np.pi * area / (perimeter ** 2)) if perimeter > 0 else 0

            # Klasifikasi ML - bukan rule-based!
            features = np.array([[aspect_ratio, circularity]])
            pred_id = pipeline.predict(features)[0]
            predicted_label = "Helm" if pred_id == 0 else "Kacamata"

    end_time = time.time()

    if show_plot:
        img_result = img.copy()
        if best_contour is not None and area > 500:
            color = (0, 200, 0) if predicted_label == "Helm" else (0, 0, 220)
            cv2.rectangle(img_result, (x, y), (x+w, y+h), color, 3)
            cv2.putText(img_result, f"{predicted_label} AR:{aspect_ratio:.2f} C:{circularity:.2f}",
                        (x, max(y-10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)); axes[0].set_title("1. Citra Asli"); axes[0].axis("off")
        axes[1].imshow(binary, cmap="gray"); axes[1].set_title("2. Citra Biner (Otsu)"); axes[1].axis("off")
        axes[2].imshow(sobel_combined, cmap="gray"); axes[2].set_title("3. Deteksi Tepi (Sobel)"); axes[2].axis("off")
        axes[3].imshow(cv2.cvtColor(img_result, cv2.COLOR_BGR2RGB)); axes[3].set_title(f"4. Hasil SVM: {predicted_label}"); axes[3].axis("off")
        plt.tight_layout(); plt.show()

    return predicted_label, aspect_ratio, circularity, end_time - start_time'''

new_classify_body = '''    # Image Processing (identik dengan extract_features saat training)
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

            # Kirim 5 fitur ke model (HARUS sama dengan saat training)
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

    return predicted_label, aspect_ratio, circularity, end_time - start_time'''

if old_classify_body in content:
    content = content.replace(old_classify_body, new_classify_body)
    print("PATCH 10 (classify_with_model) - OK")
else:
    print("PATCH 10 - GAGAL, kemungkinan sudah versi lain")

# ===================================================
# PATCH 11: Ganti semua referensi svm_pipeline -> rf_pipeline
# ===================================================
content = content.replace('svm_pipeline', 'rf_pipeline')
content = content.replace('"svm_apd_model.pkl"', '"rf_apd_model.pkl"')
content = content.replace('"hasil_evaluasi_svm.csv"', '"hasil_evaluasi_rf.csv"')
content = content.replace('RINGKASAN PERFORMA MODEL SVM', 'RINGKASAN PERFORMA MODEL RANDOM FOREST')
content = content.replace('Model SVM', 'Model Random Forest')
content = content.replace('DEMO KLASIFIKASI MENGGUNAKAN MODEL SVM', 'DEMO KLASIFIKASI MENGGUNAKAN MODEL RANDOM FOREST')
print("PATCH 11 (referensi svm->rf) - OK")

# ===================================================
# PATCH 12: Tambah StratifiedKFold dan shuffle jika belum ada
# ===================================================
if 'StratifiedKFold' not in content:
    content = content.replace(
        'from sklearn.model_selection import train_test_split, learning_curve',
        'from sklearn.model_selection import train_test_split, learning_curve, StratifiedKFold'
    )
    content = content.replace(
        '    cv=5, scoring="accuracy", n_jobs=-1',
        '    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42), scoring="accuracy", n_jobs=-1'
    )
    print("PATCH 12 (StratifiedKFold) - OK")

if 'df_train_shuffled' not in content:
    content = content.replace(
        'X = df_train[["aspect_ratio", "circularity", "solidity", "extent", "hull_ar"]].values',
        'df_train_shuffled = df_train.sample(frac=1, random_state=42).reset_index(drop=True)\nX = df_train_shuffled[["aspect_ratio", "circularity", "solidity", "extent", "hull_ar"]].values'
    )
    print("PATCH 12b (shuffle) - OK")

# Simpan hasil
with open('generate_nb.py', 'w', encoding='utf-8') as f:
    f.write(content)

print(f"\nFile baru: {len(content.splitlines())} baris")
print("Semua patch selesai! Jalankan: python generate_nb.py")
