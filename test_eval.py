import cv2
import numpy as np
import os
import glob
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

IMG_SIZE = (64, 64)

def extract_features_sobel_no_bg(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None
    gray_orig = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur_orig = cv2.GaussianBlur(gray_orig, (5, 5), 0)
    _, mask = cv2.threshold(blur_orig, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    img_rgb = cv2.bitwise_and(img, img, mask=mask)
    img_resized = cv2.resize(img_rgb, IMG_SIZE)
    img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(img_gray, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 10, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    sobelx = cv2.Sobel(binary, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(binary, cv2.CV_64F, 0, 1, ksize=3)
    sobel_combined = np.uint8(np.absolute(cv2.magnitude(sobelx, sobely)))
    return sobel_combined.flatten()

train_dir = "dataset/train"
categories = {"helm": 0, "kacamata": 1}
X_train_list = []
y_train_list = []

for label_name, label_id in categories.items():
    folder = os.path.join(train_dir, label_name)
    files = glob.glob(os.path.join(folder, "*.*"))
    for img_path in files[:150]:
        features = extract_features_sobel_no_bg(img_path)
        if features is not None:
            X_train_list.append(features)
            y_train_list.append(label_id)

X = np.array(X_train_list)
y = np.array(y_train_list)

rf_model = RandomForestClassifier(n_estimators=50, class_weight="balanced", random_state=42)
rf_model.fit(X, y)

test_dir = "dataset/test"
test_files = glob.glob(os.path.join(test_dir, "*", "*.*"))

y_true = []
y_pred = []

for img_path in test_files:
    parent = os.path.basename(os.path.dirname(img_path))
    actual = "Helm" if parent.lower() == "helm" else "Kacamata"
    y_true.append(actual)
    
    img = cv2.imread(img_path)
    if img is not None:
        features = extract_features_sobel_no_bg(img_path).reshape(1, -1)
        pred_id = rf_model.predict(features)[0]
        pred = "Helm" if pred_id == 0 else "Kacamata"
        y_pred.append(pred)
    else:
        y_pred.append("Error")

labels = ["Helm", "Kacamata"]
print(classification_report(y_true, y_pred, labels=labels, zero_division=0))
