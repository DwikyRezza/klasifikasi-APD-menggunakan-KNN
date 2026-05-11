import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score

def compare_models_three_way(csv_path):
    # 1. Load Data
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} tidak ditemukan.")
        return
        
    df = pd.read_csv(csv_path)
    print(f"Memuat {len(df)} data dari {csv_path}")
    
    # Pilih 5 fitur geometri
    features = ["aspect_ratio", "circularity", "solidity", "extent", "hull_ar"]
    X = df[features].values
    y = df["label_id"].values
    
    # 2. Definisi Model (RF, SVM, KNN)
    models = {
        "Random Forest": Pipeline([
            ("scaler", StandardScaler()),
            ("rf", RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42))
        ]),
        "SVM (RBF)": Pipeline([
            ("scaler", StandardScaler()),
            ("svm", SVC(kernel="rbf", C=1.0, probability=True, class_weight="balanced", random_state=42))
        ]),
        "KNN (k=5)": Pipeline([
            ("scaler", StandardScaler()),
            ("knn", KNeighborsClassifier(n_neighbors=5))
        ])
    }
    
    # 3. Evaluasi dengan Cross-Validation (5-Fold)
    results = []
    scoring = {
        'accuracy': 'accuracy',
        'precision': make_scorer(precision_score, pos_label=0, zero_division=0),
        'recall': make_scorer(recall_score, pos_label=0, zero_division=0),
        'f1': make_scorer(f1_score, pos_label=0, zero_division=0)
    }
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    print("\nSedang mengevaluasi 3 model (5-Fold Cross Validation)...")
    for name, pipeline in models.items():
        cv_results = cross_validate(pipeline, X, y, cv=cv, scoring=scoring)
        
        results.append({
            "Model": name,
            "Accuracy (%)": round(cv_results['test_accuracy'].mean() * 100, 2),
            "Precision (%)": round(cv_results['test_precision'].mean() * 100, 2),
            "Recall (%)": round(cv_results['test_recall'].mean() * 100, 2),
            "F1-Score (%)": round(cv_results['test_f1'].mean() * 100, 2)
        })
        print(f" - {name} selesai.")

    # 4. Tampilkan Tabel
    df_compare = pd.DataFrame(results)
    print("\n=== TABEL PERBANDINGAN MODEL (RF vs SVM vs KNN) ===")
    print(df_compare.to_string(index=False))
    
    # 5. Visualisasi
    metrics = ["Accuracy (%)", "Precision (%)", "Recall (%)", "F1-Score (%)"]
    x = np.arange(len(metrics))
    width = 0.25 # Lebar bar lebih kecil untuk muat 3 model
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    colors = ['#2196F3', '#F44336', '#4CAF50'] # Biru, Merah, Hijau
    
    for i, row in df_compare.iterrows():
        offset = (i - 1) * width
        rects = ax.bar(x + offset, row[1:], width, label=row['Model'], color=colors[i], edgecolor='black')
        
        # Tambah label angka di atas bar
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height}%',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3), 
                        textcoords="offset points",
                        ha='center', va='bottom', fontweight='bold', fontsize=9)

    ax.set_ylabel('Nilai (%)', fontsize=12)
    ax.set_title('Perbandingan Performa: RF vs SVM vs KNN', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=11)
    ax.set_ylim(0, 115)
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("perbandingan_model_3.png")
    print("\nGrafik perbandingan 3 model disimpan sebagai 'perbandingan_model_3.png'")
    plt.show()

if __name__ == "__main__":
    import os
    compare_models_three_way("features_train.csv")

