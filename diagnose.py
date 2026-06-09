import os, cv2, numpy as np, glob
from skimage.feature import graycomatrix, graycoprops, hog
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

IMG_SIZE = (128, 128)
CLASSES = ['helm', 'kacamata']

def extract_features(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    img_resized = cv2.resize(img, IMG_SIZE)
    distances = [1, 3]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    glcm = graycomatrix(img_resized, distances=distances, angles=angles,
                        levels=256, symmetric=True, normed=True)
    glcm_features = np.hstack([
        graycoprops(glcm, 'contrast').flatten(),
        graycoprops(glcm, 'correlation').flatten(),
        graycoprops(glcm, 'energy').flatten(),
        graycoprops(glcm, 'homogeneity').flatten(),
        graycoprops(glcm, 'ASM').flatten(),
        graycoprops(glcm, 'dissimilarity').flatten()
    ])
    hog_features = hog(img_resized, orientations=8, pixels_per_cell=(16, 16),
                       cells_per_block=(2, 2), visualize=False)
    return np.hstack([glcm_features, hog_features])

print("Memuat dataset...")
X, y = [], []
for ci, cn in enumerate(CLASSES):
    files = glob.glob(f'dataset/train/{cn}/*.*')
    print(f"  {cn}: {len(files)} files")
    for p in files:
        f = extract_features(p)
        if f is not None:
            X.append(f)
            y.append(ci)

X, y = np.array(X), np.array(y)
print(f"Total: {len(X)} samples, helm={sum(y==0)}, kacamata={sum(y==1)}")

scaler = StandardScaler()
Xs = scaler.fit_transform(X)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(Xs, y)

test_path = r'D:\Rezza\Gambar\Cuplikan Layar\Screenshot 2026-06-02 185948.png'
f = extract_features(test_path)
fs = scaler.transform([f])
pred = knn.predict(fs)[0]
proba = knn.predict_proba(fs)[0]
print(f"\nHasil prediksi gambar kacamata:")
print(f"  Prediksi: {CLASSES[pred]}")
print(f"  Prob helm={proba[0]:.3f}, prob kacamata={proba[1]:.3f}")

# Cek 5 tetangga terdekat
distances, indices = knn.kneighbors(fs, n_neighbors=5)
print("\n5 Tetangga Terdekat:")
for i in range(5):
    idx = indices[0][i]
    dist = distances[0][i]
    label = CLASSES[y[idx]]
    print(f"  Tetangga {i+1}: label={label}, jarak={dist:.4f}")
