import os
import cv2
from skimage.feature import graycomatrix, graycoprops
import numpy as np
import pandas as pd

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

feature_columns = [
    "contrast",
    "dissimilarity",
    "homogeneity",
    "energy",
    "h_mean",
    "s_mean",
    "v_mean",
    "h_std",
    "s_std",
    "v_std",
]

df = pd.DataFrame(data_fitur, columns=feature_columns)
df["label"] = labels

df.to_csv("hasil.csv", index=False)