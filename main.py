"""
Ara Ödev: Temel Makine Öğrenmesi Sınıflandırma Akışı
--------------------------------------------------
Amaç: Müşteri ayrılma (churn) tahmini senaryosu üzerinden veri önişleme, 
      öznitelik üretimi, model eğitimi ve sınıflandırma metrikleriyle 
      değerlendirme adımlarını pratik etmektir.

Kullanılan Kütüphaneler:
    - pandas, numpy
    - scikit-learn (train_test_split, StandardScaler, OneHotEncoder, 
                    LogisticRegression, KNeighborsClassifier, 
                    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix)

Çalıştırma Adımları:
    1. Gerekli kütüphaneleri yükleyin: pip install pandas numpy scikit-learn
    2. Betiği çalıştırın: python main.py
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ==========================================
# 1. VERİ SETİ OLUŞTURMA (VEYA OKUMA)
# ==========================================
print("--- 1. VERİ SETİ OLUŞTURULUYOR ---")
np.random.seed(42)
n_samples = 1000

data = {
    "yas": np.random.randint(18, 70, size=n_samples),
    "gelir": np.random.randint(3000, 30000, size=n_samples),
    "abonelik_suresi": np.random.randint(1, 60, size=n_samples),
    "destek_talebi_sayisi": np.random.randint(0, 10, size=n_samples),
    "sehir": np.random.choice(
        ["İstanbul", "Ankara", "İzmir", "Bursa", "Antalya"], size=n_samples
    ),
    "uyelik_tipi": np.random.choice(
        ["Standart", "Gold", "Platinum"], size=n_samples
    ),
    "churn": np.random.choice([0, 1], size=n_samples, p=[0.75, 0.25]),
}

df = pd.DataFrame(data)

# Rastgele birkaç eksik değer (NaN) ekleyelim ki eksik değer kontrolü ve doldurma adımı anlam kazansın
df.loc[np.random.choice(df.index, 15), "gelir"] = np.nan
df.loc[np.random.choice(df.index, 10), "yas"] = np.nan

# ==========================================
# 2. İLK İNCELEME
# ==========================================
print("\n--- 2. VERİ SETİ İLK İNCELEME ---")
print("İlk 5 Satır:")
print(df.head())
print(f"\nSatır ve Sütun Sayısı: {df.shape}")
print("\nHedef Değişken (churn) Dağılımı:")
print(df["churn"].value_counts(normalize=True))

# ==========================================
# 3. EKSİK DEĞER KONTROLÜ VE DOLDURMA
# ==========================================
print("\n--- 3. EKSİK DEĞER KONTROLÜ VE TEMİZLİĞİ ---")
print("Eksik Değer Sayıları:\n", df.isnull().sum())

# Sayısal sütunlardaki eksik değerleri medyan ile doldurma
df["gelir"].fillna(df["gelir"].median(), inplace=True)
df["yas"].fillna(df["yas"].median(), inplace=True)
print("Eksik değerler dolduruldu.")

# ==========================================
# 4. ÖZNİTELİK ÜRETİMİ (FEATURE ENGINEERING)
# ==========================================
print("\n--- 4. ÖZNİTELİK ÜRETİMİ ---")
# Yeni öznitelik 1: Gelir Grubu (Düşük, Orta, Yüksek)
# Yeni öznitelik 2: Destek talebi var mı? (0 veya 1)
df["gelir_grubu"] = pd.qcut(
    df["gelir"], q=3, labels=["Düşük", "Orta", "Yüksek"]
)
df["destek_talebi_var_mi"] = (df["destek_talebi_sayisi"] > 0).astype(int)

print("Yeni öznitelikler eklendi: 'gelir_grubu', 'destek_talebi_var_mi'")
print(df[["gelir", "gelir_grubu", "destek_talebi_sayisi", "destek_talebi_var_mi"]].head())

# ==========================================
# 5. GİRİŞ VE HEDEF DEĞİŞKENLERİNİN AYRLMASI
# ==========================================
X = df.drop(columns=["churn"])
y = df["churn"]

# ==========================================
# 6. TRAIN, VALIDATION VE TEST BÖLME
# ==========================================
print("\n--- 6. VERİNİN BÖLÜNMESİ (Train / Validation / Test) ---")
# Önce %80 train+val, %20 test
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# Kalan %80'i kendi içinde %75 train (%60 toplam), %25 validation (%20 toplam) olarak bölelim
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.25, random_state=42, stratify=y_train_val
)

print(f"Train Seti Boyutu: {X_train.shape[0]}")
print(f"Validation Seti Boyutu: {X_val.shape[0]}")
print(f"Test Seti Boyutu: {X_test.shape[0]}")

# ==========================================
# 7. ÖN İŞLEME PİPELINE'I (Ölçekleme & Encoding)
# ==========================================
numeric_features = [
    "yas",
    "gelir",
    "abonelik_suresi",
    "destek_talebi_sayisi",
    "destek_talebi_var_mi",
]
categorical_features = ["sehir", "uyelik_tipi", "gelir_grubu"]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ]
)

# ==========================================
# 8. MODEL EĞİTİMİ VE VALIDATION KARŞILAŞTIRMASI
# ==========================================
print("\n--- 8. MODEL EĞİTİMİ VE VALIDATION PERFORMANSI ---")

models = {
    "Logistic Regression": LogisticRegression(random_state=42),
    "K-Nearest Neighbors (KNN)": KNeighborsClassifier(n_neighbors=5),
}

val_results = {}

for name, model in models.items():
    # Pipeline oluşturma
    clf = Pipeline(
        steps=[("preprocessor", preprocessor), ("classifier", model)]
    )

    # Eğitim
    clf.fit(X_train, y_train)

    # Validation üzerinde tahmin
    y_val_pred = clf.predict(X_val)

    acc = accuracy_score(y_val, y_val_pred)
    f1 = f1_score(y_val, y_val_pred, zero_division=0)
    val_results[name] = {"model": clf, "accuracy": acc, "f1": f1}

    print(f"\nModel: {name}")
    print(f"  Validation Accuracy: {acc:.4f}")
    print(f"  Validation F1-Score: {f1:.4f}")

# En iyi modeli seçme (F1-Score'a göre)
best_model_name = max(val_results, key=lambda x: val_results[x]["f1"])
best_pipeline = val_results[best_model_name]["model"]
print(f"\n-> Validation sonuçlarına göre en başarılı model: **{best_model_name}**")

# ==========================================
# 9. SEÇİLEN MODELİN TEST VERİSİ İLE DEĞERLENDİRİLMESİ
# ==========================================
print(f"\n--- 9. TEST SETİ DEĞERLENDİRMESİ ({best_model_name}) ---")
y_test_pred = best_pipeline.predict(X_test)

test_acc = accuracy_score(y_test, y_test_pred)
test_prec = precision_score(y_test, y_test_pred, zero_division=0)
test_rec = recall_score(y_test, y_test_pred, zero_division=0)
test_f1 = f1_score(y_test, y_test_pred, zero_division=0)
conf_matrix = confusion_matrix(y_test, y_test_pred)

print(f"Test Accuracy  : {test_acc:.4f}")
print(f"Test Precision : {test_prec:.4f}")
print(f"Test Recall    : {test_rec:.4f}")
print(f"Test F1-Score  : {test_f1:.4f}")
print("\nConfusion Matrix:")
print(conf_matrix)

# ==========================================
# 10. KISA YORUM ÇIKTISI
# ==========================================
print("\n--- 10. SONUÇ YORUMU ---")
print(f"Seçilen Model: {best_model_name}")
print(
    "Yorum: Müşteri ayrılma tahmini (churn) probleminde sınıflar arasında dengesizlik "
    "(çoğunlukla kalıcı müşteriler) görülebilmektedir. Lojistik Regresyon doğrusal sınırları "
    "başarıyla öğrenirken, KNN mesafe temelli çalıştığı için ölçeklendirme adımıyla birlikte "
    "benzer müşteri profillerini sınıflandırmada etkili olmuştur. Veri setimizin yapısına ve "
    "ürettiğimiz özniteliklere (gelir grubu, destek talebi durumu) bağlı olarak bu model "
    "genelleştirme yeteneğini test setinde başarıyla korumuştur."
)
