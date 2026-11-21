import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("hasil.csv")
feature_cols = [c for c in df.columns if c != "label"]

for label_name, group_df in df.groupby("label"):
  print(f"\n{'='*40}")
  print(f"SUB-CLUSTER TEST: {label_name.upper()} (n={len(group_df)})")
  print(f"{'='*40}")

  # Scale features strictly within this label
  X_scaled = StandardScaler().fit_transform(group_df[feature_cols])

  best_k = None
  best_score = -1

  # Test splitting this label into 2, 3, 4, or 5 sub-groups
  for k in range(2, 6):
    if len(group_df) <= k:
      continue

    kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
    cluster_labels = kmeans.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, cluster_labels)

    print(f"  k={k} clusters -> Silhouette Score: {score:.3f}")

    if score > best_score:
      best_score = score
      best_k = k

  # Inspect what separates the best sub-clusters if a pattern exists
  if best_score > 0.35:
    print(f"\n  -> Notable split at k={best_k} (Score: {best_score:.3f})")
    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init="auto")
    group_df = group_df.copy()
    group_df["sub_cluster"] = kmeans.fit_predict(X_scaled)

    # Print average feature values for each sub-cluster
    print(group_df.groupby("sub_cluster")[feature_cols].mean().round(2).T)
  else:
    print("\n  -> Uniform group: No strong sub-clusters found.")
