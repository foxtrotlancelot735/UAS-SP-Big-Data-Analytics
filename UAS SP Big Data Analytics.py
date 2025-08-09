import pandas as pd
from sklearn.linear_model import LogisticRegression
import numpy as np
import statistics

# ===================================================================
# Jawaban Soal 2: Menghitung Mean, Median, dan Modus
# ===================================================================
# Data jumlah pembelian harian
pembelian_harian = [10, 20, 20, 30, 40, 50, 50, 50, 60, 70]

# Menghitung rata-rata (mean), nilai tengah (median), dan modus
mean_pembelian = statistics.mean(pembelian_harian)
median_pembelian = statistics.median(pembelian_harian)
modus_pembelian = statistics.mode(pembelian_harian)

print("--- Jawaban Soal 2 ---")
print(f"Data Pembelian: {pembelian_harian}")
print(f"Rata-rata (Mean): {mean_pembelian}")
print(f"Nilai Tengah (Median): {median_pembelian}")
print(f"Modus: {modus_pembelian}")
print("\n" + "="*50 + "\n")


# ===================================================================
# Jawaban Soal 3: Melatih Model Logistic Regression
# ===================================================================
# Data pasien yang disediakan dalam soal
data_pasien = {
    'Usia': [25, 35, 45, 30, 50],
    'Kolesterol': [200, 240, 220, 210, 250],
    'Olahraga': [1, 0, 0, 1, 1],
    'Riwayat Keluarga': [0, 1, 1, 0, 1],
    'Tekanan Darah': [120, 140, 130, 125, 145],
    'Penyakit Jantung': [0, 1, 1, 0, 1]
}

# Membuat DataFrame menggunakan pandas
df = pd.DataFrame(data_pasien)

# Memisahkan fitur target
fitur = ['Usia', 'Kolesterol', 'Olahraga', 'Riwayat Keluarga', 'Tekanan Darah']
X = df[fitur]
y = df['Penyakit Jantung']

# Membuat dan melatih model Logistic Regression
model_jantung = LogisticRegression()
model_jantung.fit(X, y)

print("--- Jawaban Soal 3 ---")
print("Model Logistic Regression untuk prediksi penyakit jantung telah berhasil dilatih.")
print("\n" + "="*50 + "\n")


# ===================================================================
# Jawaban Soal 4: Memprediksi Individu Baru
# ===================================================================
# Urutan: Usia, Kolesterol, Olahraga, Riwayat Keluarga, Tekanan Darah
individu_baru = np.array([[40, 230, 0, 1, 135]])

# Menggunakan model yang sudah dilatih untuk memprediksi
prediksi = model_jantung.predict(individu_baru)
probabilitas_prediksi = model_jantung.predict_proba(individu_baru)

print("--- Jawaban Soal 4 ---")
print(f"Data Individu Baru: {individu_baru[0]}")

# Menampilkan hasil prediksi berdasarkan nilai prediksi
if prediksi[0] == 1:
    print("Hasil Prediksi: Individu ini berisiko memiliki penyakit jantung.")
else:
    print("Hasil Prediksi: Individu ini TIDAK berisiko memiliki penyakit jantung.")

print(f"Probabilitas (tidak berisiko vs berisiko): {probabilitas_prediksi[0]}")
