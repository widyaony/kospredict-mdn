
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.dummy import DummyRegressor


# 1. MEMBACA DATASET

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(BASE_DIR, "dataset", "kos_madiun.csv")

df = pd.read_csv(DATASET_PATH)

# Mengubah fasilitas AC menjadi angka
df["ac"] = df["ac"].map({
    "ya": 1,
    "tidak": 0
})

# Menentukan fitur dan target
X = df[["jarak_km", "ac", "rating"]]
y = df["harga"]

# 1A. ANALISIS DATASET

print("\n======================================")
print(" ANALISIS DATASET")
print("======================================")

print("\nStatistik dataset:")
print(df[["jarak_km", "harga", "ac", "rating"]].describe())

print("\nKorelasi fitur dengan harga (Pearson):")
print(
    df[["jarak_km", "ac", "rating", "harga"]]
    .corr(method="pearson")["harga"]
)

print("\nKorelasi fitur dengan harga (Spearman):")
print(
    df[["jarak_km", "ac", "rating", "harga"]]
    .corr(method="spearman")["harga"]
)

print("\nRata-rata harga berdasarkan fasilitas AC:")
print(df.groupby("ac")["harga"].mean())

print("======================================")
print(" EVALUASI MODEL MACHINE LEARNING")
print("======================================")
print("Jumlah dataset:", len(df))


# 2. MENENTUKAN ALGORITMA

models = {
    "Linear Regression": LinearRegression(),

    "Decision Tree": DecisionTreeRegressor(
        max_depth=3,
        min_samples_leaf=2,
        random_state=42
    ),

    "Random Forest": RandomForestRegressor(
        n_estimators=200,
        max_depth=3,
        min_samples_leaf=2,
        random_state=42
    )
}

# Model pembanding sederhana
baseline = DummyRegressor(strategy="mean")



# 3. LEAVE-ONE-OUT CROSS VALIDATION

cv = LeaveOneOut()

hasil_evaluasi = []

for nama, model in models.items():

    # Prediksi setiap data menggunakan model yang dilatih
    # dengan 29 data lainnya
    prediksi = cross_val_predict(model, X, y, cv=cv)

    mae = mean_absolute_error(y, prediksi)
    mse = mean_squared_error(y, prediksi)
    rmse = np.sqrt(mse)
    r2 = r2_score(y, prediksi)

    hasil_evaluasi.append({
        "Model": nama,
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2 Score": r2
    })

    print("\nModel:", nama)
    print("MAE:", round(mae, 2))
    print("MSE:", round(mse, 2))
    print("RMSE:", round(rmse, 2))
    print("R2 Score:", round(r2, 4))


# 4. EVALUASI BASELINE

prediksi_baseline = cross_val_predict(
    baseline, X, y, cv=cv
)

baseline_mae = mean_absolute_error(y, prediksi_baseline)
baseline_mse = mean_squared_error(y, prediksi_baseline)
baseline_rmse = np.sqrt(baseline_mse)
baseline_r2 = r2_score(y, prediksi_baseline)

print("\n======================================")
print(" MODEL PEMBANDING (BASELINE)")
print("======================================")
print("MAE:", round(baseline_mae, 2))
print("MSE:", round(baseline_mse, 2))
print("RMSE:", round(baseline_rmse, 2))
print("R2 Score:", round(baseline_r2, 4))


# 5. MEMILIH MODEL TERBAIK

tabel = pd.DataFrame(hasil_evaluasi)
tabel = tabel.sort_values(by="MAE", ascending=True)

print("\n======================================")
print(" PERBANDINGAN HASIL MODEL")
print("======================================")
print(tabel.to_string(index=False))

model_terbaik_nama = tabel.iloc[0]["Model"]
model_terbaik = models[model_terbaik_nama]


# 5A. ANALISIS HASIL PREDIKSI SETIAP DATA

print("\n======================================")
print(" ANALISIS HASIL PREDIKSI LOOCV")
print("======================================")

# Menghasilkan prediksi untuk setiap data menggunakan LOOCV
prediksi_loocv = cross_val_predict(
    model_terbaik,
    X,
    y,
    cv=cv
)

# Membuat tabel perbandingan harga asli dan prediksi
tabel_prediksi = pd.DataFrame({
    "Nama Kos": df["nama"],
    "Harga Asli": y.values,
    "Hasil Prediksi": np.round(prediksi_loocv, 2),
    "Selisih": np.round(
        np.abs(y.values - prediksi_loocv), 2
    )
})

# Mengurutkan berdasarkan selisih terbesar
tabel_prediksi = tabel_prediksi.sort_values(
    by="Selisih",
    ascending=False
)

print("Model yang digunakan:", model_terbaik_nama)
print("\nHasil perbandingan harga:")
print(tabel_prediksi.to_string(index=False))

print("\nRata-rata selisih prediksi:",
      round(tabel_prediksi["Selisih"].mean(), 2))


# Melatih ulang model terbaik menggunakan seluruh dataset
model_terbaik.fit(X, y)

# Menyimpan model
MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "model_kos.joblib"
)

joblib.dump(model_terbaik, MODEL_PATH)

print("\n======================================")
print(" HASIL AKHIR")
print("======================================")
print("Model terbaik:", model_terbaik_nama)
print("MAE terbaik:", round(tabel.iloc[0]["MAE"], 2))
print("Baseline MAE:", round(baseline_mae, 2))

if tabel.iloc[0]["MAE"] < baseline_mae:
    print("Kesimpulan: Model terbaik mengungguli baseline.")
else:
    print("Kesimpulan: Model belum mengungguli baseline.")

print("\nModel berhasil disimpan di:")
print(MODEL_PATH)



# 6. VISUALISASI ANALISIS DATASET

# Grafik 1: Jarak kos terhadap harga
plt.figure(figsize=(7, 5))
plt.scatter(df["jarak_km"], df["harga"])
plt.title("Hubungan Jarak Kos dengan Harga")
plt.xlabel("Jarak (km)")
plt.ylabel("Harga")
plt.grid(True)
plt.tight_layout()
plt.show()


# Grafik 2: Rating kos terhadap harga
plt.figure(figsize=(7, 5))
plt.scatter(df["rating"], df["harga"])
plt.title("Hubungan Rating Kos dengan Harga")
plt.xlabel("Rating")
plt.ylabel("Harga")
plt.grid(True)
plt.tight_layout()
plt.show()


# Grafik 3: Perbandingan harga kos berdasarkan AC
plt.figure(figsize=(7, 5))

df.boxplot(
    column="harga",
    by="ac"
)

plt.title("Perbandingan Harga Kos Berdasarkan AC")
plt.suptitle("")
plt.xlabel("Fasilitas AC (0 = Tidak, 1 = Ya)")
plt.ylabel("Harga")
plt.tight_layout()
plt.show()

