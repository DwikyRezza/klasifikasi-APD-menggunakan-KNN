import cv2
import numpy as np
import os
import glob
from skimage.feature import hog
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

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

print("Mengekstrak fitur HOG data train...")
for label_name, label_id in categories.items():
    folder = os.path.join(train_dir, label_name)
    files = glob.glob(os.path.join(folder, "*.*"))
    print(f"Loading train {label_name} ({len(files)} files)...")
    for img_path in files:
        features = extract_features_hog(img_path)
        if features is not None:
            X_train_list.append(features)
            y_train_list.append(label_id)

X_train = np.array(X_train_list)
y_train = np.array(y_train_list)

# Test
test_dir = "dataset/test"
X_test_list = []
y_true_numeric = []

print("\nMengekstrak fitur HOG data test...")
for label_name, label_id in categories.items():
    folder = os.path.join(test_dir, label_name)
    files = glob.glob(os.path.join(folder, "*.*"))
    print(f"Loading test {label_name} ({len(files)} files)...")
    for img_path in files:
        features = extract_features_hog(img_path)
        if features is not None:
            X_test_list.append(features)
            y_true_numeric.append(label_id)

X_test = np.array(X_test_list)
y_true_numeric = np.array(y_true_numeric)

# Preprocessing: Scale & PCA
print("\nMelakukan standarisasi dan reduksi dimensi PCA...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

pca = PCA(n_components=100, random_state=42)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# Model KNN
print("Melatih model KNN...")
knn_model = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
knn_model.fit(X_train_pca, y_train)

# Model Random Forest
print("Melatih model Random Forest...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_pca, y_train)

# Prediksi
print("Melakukan evaluasi pada data test...")
y_pred_knn_num = knn_model.predict(X_test_pca)
y_pred_rf_num = rf_model.predict(X_test_pca)

y_true = ["Helm" if y == 0 else "Kacamata" for y in y_true_numeric]
y_pred_knn = ["Helm" if y == 0 else "Kacamata" for y in y_pred_knn_num]
y_pred_rf = ["Helm" if y == 0 else "Kacamata" for y in y_pred_rf_num]

labels = ["Helm", "Kacamata"]
print("\n=== CLASSIFICATION REPORT (HOG + PCA + KNN EVAL) ===")
print(classification_report(y_true, y_pred_knn, labels=labels, zero_division=0))

print("\n=== CLASSIFICATION REPORT (HOG + PCA + RANDOM FOREST EVAL) ===")
print(classification_report(y_true, y_pred_rf, labels=labels, zero_division=0))
