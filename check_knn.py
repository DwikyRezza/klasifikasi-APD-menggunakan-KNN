import cv2
import numpy as np
import os
import glob
from skimage.feature import hog
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

IMG_SIZE = (64, 64) 

def extract_features_hog(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None
    img_resized = cv2.resize(img, IMG_SIZE)
    img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    features = hog(img_gray, orientations=9, pixels_per_cell=(8, 8),
                   cells_per_block=(2, 2), block_norm='L2-Hys', visualize=False)
    return features

# Train
train_dir = "dataset/train"
categories = {"helm": 0, "kacamata": 1}
X_train_list = []
y_train_list = []

for label_name, label_id in categories.items():
    folder = os.path.join(train_dir, label_name)
    files = glob.glob(os.path.join(folder, "*.*"))
    for img_path in files[:150]: # Menggunakan subset 150 gambar per kelas agar proses cepat
        features = extract_features_hog(img_path)
        if features is not None:
            X_train_list.append(features)
            y_train_list.append(label_id)

X_train = np.array(X_train_list)
y_train = np.array(y_train_list)

print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")

# Preprocessing: Scale & PCA
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

pca = PCA(n_components=50, random_state=42) # 50 komponen cukup untuk subset kecil
X_train_pca = pca.fit_transform(X_train_scaled)

# Model KNN
knn_model = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
knn_model.fit(X_train_pca, y_train)

# Model Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_pca, y_train)

# Test
test_files = glob.glob(os.path.join("dataset/test", "*", "*.*"))
X_test_list = []
y_true = []

for img_path in test_files:
    parent = os.path.basename(os.path.dirname(img_path))
    actual = 0 if parent.lower() == "helm" else 1
    
    features = extract_features_hog(img_path)
    if features is not None:
        X_test_list.append(features)
        y_true.append(actual)

X_test = np.array(X_test_list)
y_true = np.array(y_true)

# Transform test data
X_test_scaled = scaler.transform(X_test)
X_test_pca = pca.transform(X_test_scaled)

y_pred_knn = knn_model.predict(X_test_pca)
y_pred_rf = rf_model.predict(X_test_pca)

print("\n=== CLASSIFICATION REPORT (HOG + PCA + KNN QUICK CHECK) ===")
print(classification_report(y_true, y_pred_knn, target_names=["Helm", "Kacamata"]))

print("\n=== CLASSIFICATION REPORT (HOG + PCA + RANDOM FOREST QUICK CHECK) ===")
print(classification_report(y_true, y_pred_rf, target_names=["Helm", "Kacamata"]))
