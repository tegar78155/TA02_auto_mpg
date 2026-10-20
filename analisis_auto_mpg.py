
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# ==========================================
# 1. MEMBACA DATASET AUTO MPG
# ==========================================

kolom = [
    "mpg",
    "cylinders",
    "displacement",
    "horsepower",
    "weight",
    "acceleration",
    "model_year",
    "origin",
    "car_name"
]

data = []

with open("auto-mpg.data", "r", encoding="utf-8") as file:
    for baris in file:
        baris = baris.strip()

        if not baris:
            continue

        # Pisahkan 8 kolom pertama dan sisakan
        # nama mobil sebagai satu bagian utuh
        bagian = re.split(r"\s+", baris, maxsplit=8)

        if len(bagian) == 9:
            bagian[8] = bagian[8].strip().strip('"')
            data.append(bagian)

df = pd.DataFrame(data, columns=kolom)

# Ubah kolom numerik menjadi angka
kolom_numerik = kolom[:-1]

for nama in kolom_numerik:
    df[nama] = pd.to_numeric(
        df[nama].replace("?", np.nan),
        errors="coerce"
    )

print("=" * 50)
print("ANALISIS DATASET AUTO MPG")
print("=" * 50)

# ==========================================
# 2. MENAMPILKAN DATA DAN INFORMASI DATASET
# ==========================================

print("\n=== LIMA BARIS PERTAMA ===")
print(df.head().to_string(index=False))

print("\n=== NAMA KOLOM ===")
print(df.columns.tolist())

print("\n=== INFORMASI DATASET ===")
df.info()

print("\nUkuran dataset:", df.shape)

# ==========================================
# 3. PEMERIKSAAN NILAI KOSONG
# ==========================================

print("\n=== JUMLAH NILAI KOSONG ===")
print(df.isnull().sum())

# ==========================================
# 4. STATISTIK DESKRIPTIF
# ==========================================

print("\n=== STATISTIK BERAT DAN EFISIENSI BBM ===")
print(df[["weight", "mpg"]].describe())

# ==========================================
# 5. PEMBERSIHAN DATA
# ==========================================

# Model hanya menggunakan weight dan mpg
data_model = df[["weight", "mpg"]].dropna().copy()

print("\n=== DATA SETELAH PEMBERSIHAN ===")
print("Jumlah data untuk model:", len(data_model))
print("Jumlah nilai kosong:", data_model.isnull().sum().sum())

# ==========================================
# 6. VISUALISASI DATA
# ==========================================

plt.figure(figsize=(8, 5))

plt.scatter(
    data_model["weight"],
    data_model["mpg"],
    alpha=0.7
)

plt.title("Hubungan Berat Kendaraan dengan Efisiensi BBM")
plt.xlabel("Berat Kendaraan (pound)")
plt.ylabel("Efisiensi Bahan Bakar (MPG)")
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig("scatterplot_weight_mpg.png", dpi=300)
plt.close()

print("\nGrafik tersimpan: scatterplot_weight_mpg.png")

# ==========================================
# 7. MEMISAHKAN VARIABEL X DAN y
# ==========================================

X = data_model[["weight"]]
y = data_model["mpg"]

# Membagi data: 80% training dan 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\n=== PEMBAGIAN DATA ===")
print("Data training:", len(X_train))
print("Data testing :", len(X_test))

# ==========================================
# 8. MELATIH MODEL REGRESI LINEAR
# ==========================================

model = LinearRegression()
model.fit(X_train, y_train)

# Prediksi data testing
y_pred = model.predict(X_test)

# ==========================================
# 9. EVALUASI MODEL
# ==========================================

r2 = r2_score(y_test, y_pred)

rmse = np.sqrt(
    mean_squared_error(y_test, y_pred)
)

mae = np.mean(np.abs(y_test - y_pred))

print("\n" + "=" * 50)
print("HASIL EVALUASI MODEL REGRESI LINEAR")
print("=" * 50)

print(f"Intercept (konstanta): {model.intercept_:.4f}")
print(f"Koefisien weight     : {model.coef_[0]:.6f}")
print(f"R-squared (R2)       : {r2:.4f}")
print(f"RMSE                 : {rmse:.4f} MPG")
print(f"MAE                  : {mae:.4f} MPG")

print("\nPersamaan regresi:")
print(
    f"MPG = {model.intercept_:.4f} "
    f"+ ({model.coef_[0]:.6f} x berat kendaraan)"
)

# ==========================================
# 10. CONTOH PREDIKSI
# ==========================================

berat_baru = pd.DataFrame({"weight": [3000]})
hasil_prediksi = model.predict(berat_baru)[0]

print("\n=== CONTOH PREDIKSI ===")
print("Berat kendaraan: 3000 pound")
print(f"Prediksi efisiensi: {hasil_prediksi:.2f} MPG")

# ==========================================
# 11. MENYIMPAN MODEL
# ==========================================

joblib.dump(model, "model_regresi_auto_mpg.joblib")

print("\nModel berhasil disimpan:")
print("model_regresi_auto_mpg.joblib")

print("\n=== PROGRAM SELESAI ===")
