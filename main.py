import os
import cv2
import joblib
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from skimage.feature import graycomatrix, graycoprops

# 1. Tentukan direktori dataset latihanmu
dataset_dir = "C:/Users/user/Downloads/skin_type_classification_dataset/train"
categories = ["normal", "oily", "dry"]

data_fitur = []
labels = []


def proses_ekstraksi_lanjutan(image_path):
  # Baca gambar
  img = cv2.imread(image_path)
  if img is None:
    return None

  h, w, _ = img.shape

  # Preprocessing: Crop ROI di area tengah wajah/pipi
  ymin, ymax = int(h * 0.3), int(h * 0.7)
  xmin, xmax = int(w * 0.3), int(w * 0.7)
  roi = img[ymin:ymax, xmin:xmax]

  # Ekstrak Fitur GLCM (Tekstur)
  gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
  glcm = graycomatrix(
      gray,
      distances=[1, 2],
      angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
      levels=256,
      symmetric=True,
      normed=True,
  )
  contrast = graycoprops(glcm, "contrast").mean()
  dissimilarity = graycoprops(glcm, "dissimilarity").mean()
  homogeneity = graycoprops(glcm, "homogeneity").mean()
  energy = graycoprops(glcm, "energy").mean()

  # Ekstrak Fitur Warna HSV (Mean & Std Dev)
  hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
  h_mean, s_mean, v_mean = (
      hsv[:, :, 0].mean(),
      hsv[:, :, 1].mean(),
      hsv[:, :, 2].mean(),
  )
  h_std, s_std, v_std = hsv[:, :, 0].std(), hsv[:, :, 1].std(), hsv[:, :, 2].std()

  return [
      contrast,
      dissimilarity,
      homogeneity,
      energy,
      h_mean,
      s_mean,
      v_mean,
      h_std,
      s_std,
      v_std,
  ]


# 2. Looping membaca SELURUH data gambar di dataset
print("Sedang memproses seluruh dataset untuk pelatihan model...")
for category in categories:
  folder_path = os.path.join(dataset_dir, category)
  if not os.path.exists(folder_path):
    continue

  files = os.listdir(folder_path)
  for file in files:
    file_path = os.path.join(folder_path, file)
    fitur = proses_ekstraksi_lanjutan(file_path)

    if fitur is not None:
      data_fitur.append(fitur)
      labels.append(category)

print(f"Selesai ekstraksi! Total keseluruhan data fitur: {len(data_fitur)}")

# 3. Membagi data Latih & Uji
X_train, X_test, y_train, y_test = train_test_split(
    data_fitur, labels, test_size=0.2, random_state=42, stratify=labels
)

# 4. Normalisasi dan Training Model K-NN
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

knn = KNeighborsClassifier(n_neighbors=5, weights="distance")
knn.fit(X_train_scaled, y_train)

# Evaluasi Performa
y_pred = knn.predict(X_test_scaled)
akurasi = accuracy_score(y_test, y_pred)
print(f"\nAkurasi Optimal Model: {akurasi * 100:.2f}%")
print("\nLaporan Klasifikasi:")
print(classification_report(y_test, y_pred))

# 5. Menyimpan Model dan Scaler ke file (Supaya aplikasinya 'jadi' dan permanen)
joblib.dump(knn, "model_skin_knn.pkl")
joblib.dump(scaler, "scaler.pkl")
print(
    "Model K-NN dan Scaler berhasil disimpan ('model_skin_knn.pkl' &"
    " 'scaler.pkl')!"
)


# 6. Fungsi untuk Memprediksi Foto Wajah Baru
def prediksi_wajah_baru(image_path):
  if not os.path.exists(image_path):
    print(f"\n[Perhatian] File gambar '{image_path}' tidak ditemukan.")
    return

  # Muat kembali model dan scaler yang sudah disimpan
  loaded_knn = joblib.load("model_skin_knn.pkl")
  loaded_scaler = joblib.load("scaler.pkl")

  # Ekstrak fitur dari foto baru
  fitur_baru = proses_ekstraksi_lanjutan(image_path)
  if fitur_baru is None:
    print("Gagal membaca gambar baru.")
    return

  # Normalisasi fitur baru menggunakan scaler yang sama
  fitur_scaled = loaded_scaler.transform([fitur_baru])

  # Lakukan prediksi
  prediksi = loaded_knn.predict(fitur_scaled)[0]

  print(f"\n==========================================")
  print(f" HASIL PREDIKSI UNTUK FILE: {image_path}")
  print(f" Jenis Kulit Wajah Terdeteksi : {prediksi.upper()}")
  print(f"==========================================")


# 7. Menguji fungsi prediksi menggunakan file 'wajah.jpg' di folder proyekmu
prediksi_wajah_baru("wajah.jpg")