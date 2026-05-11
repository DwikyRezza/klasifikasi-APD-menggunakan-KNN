import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

def run_advanced_comparison(csv_path):
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} tidak ditemukan.")
        return

    df = pd.read_csv(csv_path)
    print(f"Memuat {len(df)} data dari {csv_path}")

    # Outlier Removal (mengikuti logika terbaik di generate_nb.py)
    Q1 = df[['aspect_ratio', 'circularity', 'solidity']].quantile(0.05)
    Q3 = df[['aspect_ratio', 'circularity', 'solidity']].quantile(0.95)
    IQR = Q3 - Q1
    df_clean = df[~((df[['aspect_ratio', 'circularity', 'solidity']] < (Q1 - 1.5 * IQR)) | (df[['aspect_ratio', 'circularity', 'solidity']] > (Q3 + 1.5 * IQR))).any(axis=1)]
    print(f"Data bersih setelah hapus outlier: {len(df_clean)}")

    features = ["aspect_ratio", "circularity", "solidity", "extent", "hull_ar"]
    X = df_clean[features].values
    y = df_clean["label_id"].values

    # Split Data (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Definisi Pipeline dan Parameter Grid
    pipelines = {
        "Random Forest": Pipeline([
            ("scaler", StandardScaler()),
            ("rf", RandomForestClassifier(random_state=42, class_weight="balanced"))
        ]),
        "SVM (RBF)": Pipeline([
            ("scaler", StandardScaler()),
            ("svm", SVC(probability=True, random_state=42, class_weight="balanced"))
        ]),
        "KNN": Pipeline([
            ("scaler", StandardScaler()),
            ("knn", KNeighborsClassifier())
        ])
    }

    param_grids = {
        "Random Forest": {
            'rf__n_estimators': [100, 200, 300],
            'rf__max_depth': [None, 10, 20],
            'rf__min_samples_split': [2, 5, 10]
        },
        "SVM (RBF)": {
            'svm__C': [0.1, 1, 10, 100],
            'svm__gamma': ['scale', 'auto', 0.1, 1]
        },
        "KNN": {
            'knn__n_neighbors': [3, 5, 7, 9],
            'knn__weights': ['uniform', 'distance'],
            'knn__metric': ['euclidean', 'manhattan']
        }
    }

    best_models = {}
    results = []
    y_preds = {}

    print("\n--- MULAI HYPERPARAMETER TUNING ---")
    for name in pipelines.keys():
        print(f"Tuning {name}...")
        grid = GridSearchCV(pipelines[name], param_grids[name], cv=5, scoring='accuracy', n_jobs=-1)
        grid.fit(X_train, y_train)
        
        best_models[name] = grid.best_estimator_
        print(f"  Best Params: {grid.best_params_}")
        
        # Predict on test set
        y_pred = grid.predict(X_test)
        y_preds[name] = y_pred
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, pos_label=0, zero_division=0)
        rec = recall_score(y_test, y_pred, pos_label=0, zero_division=0)
        f1 = f1_score(y_test, y_pred, pos_label=0, zero_division=0)
        
        results.append({
            "Model": name,
            "Accuracy (%)": round(acc * 100, 2),
            "Precision (%)": round(prec * 100, 2),
            "Recall (%)": round(rec * 100, 2),
            "F1-Score (%)": round(f1 * 100, 2)
        })

    # Simpan model RF terbaik (karena biasanya yang terbaik)
    joblib.dump(best_models["Random Forest"], "rf_apd_model_tuned.pkl")
    print("\nModel Random Forest terbaik disimpan sebagai 'rf_apd_model_tuned.pkl'")

    # --- 1. TABEL PERBANDINGAN ---
    df_compare = pd.DataFrame(results)
    print("\n=== TABEL HASIL EVALUASI (TEST SET) ===")
    print(df_compare.to_string(index=False))

    # --- 2. GRAFIK PERBANDINGAN METRIK ---
    metrics = ["Accuracy (%)", "Precision (%)", "Recall (%)", "F1-Score (%)"]
    x = np.arange(len(metrics))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(14, 8))
    colors = ['#2196F3', '#F44336', '#4CAF50']
    
    for i, row in df_compare.iterrows():
        offset = (i - 1) * width
        rects = ax.bar(x + offset, row[1:], width, label=row['Model'], color=colors[i], edgecolor='black')
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height}%', xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontweight='bold', fontsize=9)

    ax.set_ylabel('Nilai (%)', fontsize=12)
    ax.set_title('Perbandingan Performa Setelah Tuning (Test Set)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=11)
    ax.set_ylim(0, 115)
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig("adv_comparison_metrics.png")
    print("Grafik metrik disimpan ke 'adv_comparison_metrics.png'")
    plt.close()

    # --- 3. CONFUSION MATRIX ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    class_names = ["Helm", "Kacamata"]
    
    for idx, name in enumerate(pipelines.keys()):
        cm = confusion_matrix(y_test, y_preds[name])
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
        disp.plot(ax=axes[idx], cmap='Blues', colorbar=False)
        axes[idx].set_title(f"Confusion Matrix: {name}", fontweight='bold')
    
    plt.tight_layout()
    plt.savefig("adv_comparison_confusion_matrix.png")
    print("Grafik Confusion Matrix disimpan ke 'adv_comparison_confusion_matrix.png'")
    plt.close()

    # --- 4. FEATURE IMPORTANCE (Untuk Random Forest) ---
    rf_best = best_models["Random Forest"].named_steps['rf']
    importances = rf_best.feature_importances_
    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(10, 6))
    plt.title("Feature Importance - Random Forest (Tuned)", fontsize=14, fontweight='bold')
    plt.bar(range(X.shape[1]), importances[indices], align="center", color="#FF9800", edgecolor='black')
    plt.xticks(range(X.shape[1]), [features[i] for i in indices], rotation=15, fontsize=12)
    plt.ylabel("Skor Kepentingan", fontsize=12)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig("adv_feature_importance.png")
    print("Grafik Feature Importance disimpan ke 'adv_feature_importance.png'")
    plt.close()

    print("\nSemua proses analisis selesai!")

if __name__ == "__main__":
    run_advanced_comparison("features_train.csv")
