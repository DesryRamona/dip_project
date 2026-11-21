import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("hasil.csv")
feature_cols = [c for c in df.columns if c != "label"]

for label_name, group_df in df.groupby("label"):
  print(f"\n{'='*40}")
  print(f"ANALYSIS FOR LABEL: {label_name.upper()} (n={len(group_df)})")
  print(f"{'='*40}")

  X_sub = group_df[feature_cols]

  # 1. Correlation Matrix (Linear connections between pairs of features)
  corr = X_sub.corr()
  # Print only strong correlations (above 0.6 or below -0.6, excluding self-correlation)
  print("Strong Feature Pairs (|r| > 0.6):")
  found_strong = False
  for i in range(len(feature_cols)):
    for j in range(i + 1, len(feature_cols)):
      r = corr.iloc[i, j]
      if abs(r) > 0.6:
        print(f"  {feature_cols[i]} <-> {feature_cols[j]}: {r:.2f}")
        found_strong = True
  if not found_strong:
    print("  None found.")

  # 2. Scale & Run PCA inside this single class
  X_scaled = StandardScaler().fit_transform(X_sub)
  pca = PCA(n_components=3)
  pca.fit(X_scaled)

  var_ratios = [f"{v:.1%}" for v in pca.explained_variance_ratio_]
  print(f"\nTop 3 PCA Variance Explained: {', '.join(var_ratios)}")
