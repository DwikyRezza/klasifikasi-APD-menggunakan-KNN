import cv2
import numpy as np
import os
import glob
import matplotlib.pyplot as plt

IMG_SIZE = (64, 64) 

def extract_features_sobel_no_bg(image_path, output_name):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error loading {image_path}")
        return None

    gray_orig = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    blur_orig = cv2.GaussianBlur(gray_orig, (5, 5), 0)
    
    # Background removal using OTSU
    _, mask = cv2.threshold(blur_orig, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    kernel = np.ones((5,5), np.uint8)
    mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    img_rgb = cv2.bitwise_and(img, img, mask=mask_closed)

    img_resized = cv2.resize(img_rgb, IMG_SIZE)

    img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(img_gray, (5, 5), 0)

    _, binary = cv2.threshold(blurred, 10, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    sobelx = cv2.Sobel(binary, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(binary, cv2.CV_64F, 0, 1, ksize=3)
    sobel_combined = np.uint8(np.absolute(cv2.magnitude(sobelx, sobely)))
    
    fig, axes = plt.subplots(1, 5, figsize=(15, 3))
    axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Original")
    axes[1].imshow(mask_closed, cmap='gray')
    axes[1].set_title("Mask Closed")
    axes[2].imshow(cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB))
    axes[2].set_title("Background Removed")
    axes[3].imshow(binary, cmap='gray')
    axes[3].set_title("Binary (resized)")
    axes[4].imshow(sobel_combined, cmap='gray')
    axes[4].set_title("Sobel Edge")
    
    for ax in axes:
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_name)
    plt.close()

    features = sobel_combined.flatten()
    print(f"{output_name}: sum={np.sum(sobel_combined)}, non_zero={np.count_nonzero(sobel_combined)}")
    return features

test_helm = glob.glob("dataset/test/helm/*.*")[:3]
test_kacamata = glob.glob("dataset/test/kacamata/*.*")[:3]

for i, p in enumerate(test_helm):
    extract_features_sobel_no_bg(p, f"scratch/helm_{i}.png")
    
for i, p in enumerate(test_kacamata):
    extract_features_sobel_no_bg(p, f"scratch/kacamata_{i}.png")
