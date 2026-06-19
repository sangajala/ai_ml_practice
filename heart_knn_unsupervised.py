import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors, LocalOutlierFactor
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score

RANDOM_STATE = 42
K = 5  # number of neighbours

# ── 1. Load & preprocess (no target label used) ───────────────────────────────
df = pd.read_csv("heart (1).csv")
print(f"Dataset: {df.shape[0]} rows, {df.shape[1]} columns")

# Outlier removal via IQR
numeric_cols = ["Age", "RestingBP", "Cholesterol", "MaxHR", "Oldpeak"]
df_clean = df.copy()
for col in numeric_cols:
    Q1, Q3 = df_clean[col].quantile([0.25, 0.75])
    IQR = Q3 - Q1
    df_clean = df_clean[
        (df_clean[col] >= Q1 - 1.5 * IQR) &
        (df_clean[col] <= Q3 + 1.5 * IQR)
    ]
df_clean = df_clean.reset_index(drop=True)
print(f"After outlier removal: {len(df_clean)} rows\n")

# Encode categoricals — HeartDisease dropped (unsupervised)
cat_cols = ["Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope"]
df_enc = pd.get_dummies(df_clean, columns=cat_cols, drop_first=True)
X_raw = df_enc.drop("HeartDisease", axis=1)

scaler = StandardScaler()
X = scaler.fit_transform(X_raw)

# PCA → 2D for all visualisations
pca = PCA(n_components=2, random_state=RANDOM_STATE)
X_2d = pca.fit_transform(X)
var = pca.explained_variance_ratio_ * 100
print(f"PCA variance explained: PC1={var[0]:.1f}%  PC2={var[1]:.1f}%\n")

# ─────────────────────────────────────────────────────────────────────────────
# PART A — KNN: K-Distance Graph (helps choose eps for DBSCAN)
# ─────────────────────────────────────────────────────────────────────────────
print("=== PART A: K-Distance Graph ===")
nbrs = NearestNeighbors(n_neighbors=K)
nbrs.fit(X)
distances, indices = nbrs.kneighbors(X)
k_distances = np.sort(distances[:, K - 1])[::-1]

plt.figure(figsize=(9, 4))
plt.plot(k_distances, color="#003087", linewidth=1.5)
plt.axhline(y=np.percentile(k_distances, 10), color="#c62828",
            linestyle="--", linewidth=1.5, label="Suggested eps (10th pct)")
plt.title(f"K-Distance Graph  (K={K})  — Helps choose DBSCAN eps",
          fontsize=12, fontweight="bold")
plt.xlabel("Points (sorted by distance)")
plt.ylabel(f"Distance to {K}th Nearest Neighbour")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

eps_value = round(float(np.percentile(k_distances, 10)), 3)
print(f"Suggested eps: {eps_value}\n")

# ─────────────────────────────────────────────────────────────────────────────
# PART B — KNN Anomaly Detection (avg distance to K neighbours)
# ─────────────────────────────────────────────────────────────────────────────
print("=== PART B: KNN Anomaly Detection ===")
knn_avg_dist = distances[:, 1:].mean(axis=1)          # exclude self (index 0)
threshold_95 = float(np.percentile(knn_avg_dist, 95))
anomaly_knn  = knn_avg_dist > threshold_95

print(f"Anomaly threshold (95th pct): {threshold_95:.3f}")
print(f"Anomalies detected: {anomaly_knn.sum()} / {len(df_clean)}")
print(f"Avg age  — anomaly: {df_clean.loc[anomaly_knn, 'Age'].mean():.1f}  "
      f"normal: {df_clean.loc[~anomaly_knn, 'Age'].mean():.1f}")
print(f"HeartDisease rate — anomaly: "
      f"{df_clean.loc[anomaly_knn, 'HeartDisease'].mean():.2f}  "
      f"normal: {df_clean.loc[~anomaly_knn, 'HeartDisease'].mean():.2f}\n")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Scatter
axes[0].scatter(X_2d[~anomaly_knn, 0], X_2d[~anomaly_knn, 1],
                c="#005EB8", alpha=0.45, s=25, label="Normal")
axes[0].scatter(X_2d[anomaly_knn, 0],  X_2d[anomaly_knn, 1],
                c="#c62828", alpha=0.9, s=60, marker="X", label="Anomaly (KNN)")
axes[0].set_title("KNN Anomaly Detection", fontweight="bold")
axes[0].set_xlabel(f"PC1 ({var[0]:.1f}%)")
axes[0].set_ylabel(f"PC2 ({var[1]:.1f}%)")
axes[0].legend(); axes[0].grid(alpha=0.3)

# Distance histogram
axes[1].hist(knn_avg_dist[~anomaly_knn], bins=35, color="#005EB8",
             alpha=0.7, label="Normal", edgecolor="white")
axes[1].hist(knn_avg_dist[anomaly_knn],  bins=10, color="#c62828",
             alpha=0.85, label="Anomaly", edgecolor="white")
axes[1].axvline(threshold_95, color="#f9a825", linewidth=2,
                linestyle="--", label=f"Threshold ({threshold_95:.2f})")
axes[1].set_title("KNN Avg-Distance Distribution", fontweight="bold")
axes[1].set_xlabel("Avg Distance to 5 Nearest Neighbours")
axes[1].set_ylabel("Count"); axes[1].legend(); axes[1].grid(alpha=0.3)

plt.suptitle("Part B — KNN Anomaly Detection", fontsize=13, fontweight="bold")
plt.tight_layout(); plt.show()

# ─────────────────────────────────────────────────────────────────────────────
# PART C — Local Outlier Factor (KNN-based density estimation)
# ─────────────────────────────────────────────────────────────────────────────
print("=== PART C: Local Outlier Factor (LOF) ===")
lof = LocalOutlierFactor(n_neighbors=K, contamination=0.05)
lof_labels  = lof.fit_predict(X)        # -1 = outlier, 1 = inlier
lof_scores  = -lof.negative_outlier_factor_
anomaly_lof = lof_labels == -1

print(f"LOF outliers detected: {anomaly_lof.sum()} / {len(df_clean)}")
print(f"HeartDisease rate — LOF outlier: "
      f"{df_clean.loc[anomaly_lof, 'HeartDisease'].mean():.2f}  "
      f"inlier: {df_clean.loc[~anomaly_lof, 'HeartDisease'].mean():.2f}\n")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Scatter
axes[0].scatter(X_2d[~anomaly_lof, 0], X_2d[~anomaly_lof, 1],
                c="#003087", alpha=0.45, s=25, label="Inlier")
axes[0].scatter(X_2d[anomaly_lof, 0],  X_2d[anomaly_lof, 1],
                c="#c62828", alpha=0.9, s=60, marker="X", label="LOF Outlier")
axes[0].set_title("Local Outlier Factor", fontweight="bold")
axes[0].set_xlabel(f"PC1 ({var[0]:.1f}%)")
axes[0].set_ylabel(f"PC2 ({var[1]:.1f}%)")
axes[0].legend(); axes[0].grid(alpha=0.3)

# LOF score scatter (bubble size = score)
sc = axes[1].scatter(X_2d[:, 0], X_2d[:, 1],
                     c=lof_scores, cmap="RdYlGn_r",
                     s=lof_scores * 12, alpha=0.6)
plt.colorbar(sc, ax=axes[1], label="LOF Score (higher = more outlier-like)")
axes[1].set_title("LOF Score Map", fontweight="bold")
axes[1].set_xlabel(f"PC1 ({var[0]:.1f}%)")
axes[1].set_ylabel(f"PC2 ({var[1]:.1f}%)")
axes[1].grid(alpha=0.3)

plt.suptitle("Part C — Local Outlier Factor (KNN-based)", fontsize=13, fontweight="bold")
plt.tight_layout(); plt.show()

# ─────────────────────────────────────────────────────────────────────────────
# PART D — DBSCAN clustering (KNN-derived eps)
# ─────────────────────────────────────────────────────────────────────────────
print("=== PART D: DBSCAN Clustering (KNN-derived eps) ===")
db = DBSCAN(eps=eps_value, min_samples=K)
db_labels = db.fit_predict(X)

n_clusters = len(set(db_labels)) - (1 if -1 in db_labels else 0)
n_noise    = (db_labels == -1).sum()
print(f"Clusters found: {n_clusters}  |  Noise points: {n_noise}")

if n_clusters > 1:
    sil = silhouette_score(X[db_labels != -1], db_labels[db_labels != -1])
    print(f"Silhouette score (excl. noise): {sil:.3f}")

for c in sorted(set(db_labels)):
    mask = db_labels == c
    label = f"Cluster {c}" if c != -1 else "Noise"
    hd = df_clean.loc[mask, "HeartDisease"].mean()
    print(f"  {label:12s}  n={mask.sum():4d}  HeartDisease rate={hd:.2f}")
print()

palette   = sns.color_palette("Set1", max(db_labels) + 2)
noise_col = "#aaaaaa"

plt.figure(figsize=(8, 6))
for c in sorted(set(db_labels)):
    mask  = db_labels == c
    color = noise_col if c == -1 else palette[c % len(palette)]
    label = f"Cluster {c}  (n={mask.sum()})" if c != -1 else f"Noise  (n={mask.sum()})"
    plt.scatter(X_2d[mask, 0], X_2d[mask, 1],
                color=color, alpha=0.65 if c != -1 else 0.35,
                s=35 if c != -1 else 15, label=label)

plt.title(f"DBSCAN Clustering  (eps={eps_value}, min_samples={K})",
          fontsize=12, fontweight="bold")
plt.xlabel(f"PC1 ({var[0]:.1f}% variance)")
plt.ylabel(f"PC2 ({var[1]:.1f}% variance)")
plt.legend(); plt.grid(alpha=0.3)
plt.tight_layout(); plt.show()

# ─────────────────────────────────────────────────────────────────────────────
# PART E — KNN Neighbourhood Profile (who are your neighbours?)
# ─────────────────────────────────────────────────────────────────────────────
print("=== PART E: KNN Neighbourhood Profiles ===")
nbrs5 = NearestNeighbors(n_neighbors=K + 1)
nbrs5.fit(X)
_, idx = nbrs5.kneighbors(X)

neighbour_hd = []
for i in range(len(df_clean)):
    neighbours = idx[i, 1:]      # exclude self
    rate = df_clean.loc[neighbours, "HeartDisease"].mean()
    neighbour_hd.append(rate)

df_clean["Neighbour_HD_Rate"] = neighbour_hd

print("Avg neighbour heart disease rate by actual HeartDisease label:")
print(df_clean.groupby("HeartDisease")["Neighbour_HD_Rate"].mean().round(3))
print()

plt.figure(figsize=(7, 4))
for label, color, name in [(0, "#005EB8", "No Heart Disease"),
                            (1, "#c62828", "Heart Disease")]:
    vals = df_clean[df_clean["HeartDisease"] == label]["Neighbour_HD_Rate"]
    plt.hist(vals, bins=20, color=color, alpha=0.6, label=name, edgecolor="white")
plt.title(f"KNN Neighbourhood Heart Disease Rate (K={K})",
          fontsize=12, fontweight="bold")
plt.xlabel("Avg Heart Disease Rate Among K Nearest Neighbours")
plt.ylabel("Count"); plt.legend(); plt.grid(alpha=0.3)
plt.tight_layout(); plt.show()

print("Done.")
