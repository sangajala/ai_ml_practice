import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score

RANDOM_STATE = 42

# ── 1. Load & preprocess (no target label used) ───────────────────────────────
df = pd.read_csv("heart (1).csv")
print(f"Dataset shape: {df.shape}")

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
print(f"Rows after outlier removal: {len(df_clean)}\n")

# Encode categoricals — drop HeartDisease (unsupervised: no labels)
cat_cols = ["Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope"]
df_encoded = pd.get_dummies(df_clean, columns=cat_cols, drop_first=True)
df_features = df_encoded.drop("HeartDisease", axis=1)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_features)

# ── 2. Find optimal K for K-Means (Elbow + Silhouette) ────────────────────────
print("Finding optimal number of clusters ...")
inertias, silhouettes = [], []
K_range = range(2, 9)

for k in K_range:
    km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    labels = km.fit_predict(X_scaled)
    inertias.append(km.inertia_)
    silhouettes.append(silhouette_score(X_scaled, labels))

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(list(K_range), inertias, "o-", color="#005EB8", linewidth=2)
axes[0].set_title("Elbow Method — Inertia vs K")
axes[0].set_xlabel("Number of Clusters (K)")
axes[0].set_ylabel("Inertia")
axes[0].grid(alpha=0.3)

axes[1].plot(list(K_range), silhouettes, "o-", color="#003087", linewidth=2)
axes[1].set_title("Silhouette Score vs K")
axes[1].set_xlabel("Number of Clusters (K)")
axes[1].set_ylabel("Silhouette Score")
axes[1].grid(alpha=0.3)

plt.suptitle("K-Means: Optimal Cluster Selection", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.show()

best_k = list(K_range)[silhouettes.index(max(silhouettes))]
print(f"Best K by silhouette: {best_k}  (score: {max(silhouettes):.3f})\n")

# ── 3. K-Means clustering with best K ─────────────────────────────────────────
km_final = KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init=10)
cluster_labels = km_final.fit_predict(X_scaled)
df_clean = df_clean.copy()
df_clean["Cluster"] = cluster_labels

print("=== Cluster Summary ===")
summary = df_clean.groupby("Cluster")[
    ["Age", "RestingBP", "Cholesterol", "MaxHR", "Oldpeak", "HeartDisease"]
].mean().round(2)
print(summary.to_string())
print()

# ── 4. PCA → 2D for cluster visualisation ─────────────────────────────────────
pca = PCA(n_components=2, random_state=RANDOM_STATE)
X_pca = pca.fit_transform(X_scaled)
explained = pca.explained_variance_ratio_ * 100

plt.figure(figsize=(8, 6))
palette = sns.color_palette("Set1", best_k)
for c in range(best_k):
    mask = cluster_labels == c
    plt.scatter(X_pca[mask, 0], X_pca[mask, 1],
                label=f"Cluster {c}  (n={mask.sum()})",
                color=palette[c], alpha=0.65, s=40)

plt.title(f"K-Means Clusters (K={best_k}) — PCA Projection", fontsize=13, fontweight="bold")
plt.xlabel(f"PC1 ({explained[0]:.1f}% variance)")
plt.ylabel(f"PC2 ({explained[1]:.1f}% variance)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# ── 5. Cluster profiles ────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
axes = axes.flatten()
profile_cols = ["Age", "RestingBP", "Cholesterol", "MaxHR", "Oldpeak", "HeartDisease"]

for i, col in enumerate(profile_cols):
    data = [df_clean[df_clean["Cluster"] == c][col].values for c in range(best_k)]
    axes[i].boxplot(data, patch_artist=True,
                    boxprops=dict(facecolor="#d6e8f7", color="#003087"),
                    medianprops=dict(color="#c62828", linewidth=2))
    axes[i].set_xticklabels([f"C{c}" for c in range(best_k)])
    axes[i].set_title(col, fontweight="bold")
    axes[i].grid(axis="y", alpha=0.3)

plt.suptitle("Cluster Profiles — Feature Distribution", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.show()

# ── 6. KNN anomaly detection ───────────────────────────────────────────────────
print("=== KNN Anomaly Detection ===")
K_NEIGHBOURS = 5
nbrs = NearestNeighbors(n_neighbors=K_NEIGHBOURS + 1)  # +1 because point is own neighbour
nbrs.fit(X_scaled)
distances, _ = nbrs.kneighbors(X_scaled)

# Average distance to k nearest neighbours (excluding self at index 0)
knn_dist = distances[:, 1:].mean(axis=1)
threshold = np.percentile(knn_dist, 95)   # top 5% = anomalies
anomaly_mask = knn_dist > threshold

print(f"KNN avg-distance threshold (95th pct): {threshold:.3f}")
print(f"Anomalies detected: {anomaly_mask.sum()} / {len(df_clean)} patients\n")

# Anomaly scatter on PCA space
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[~anomaly_mask, 0], X_pca[~anomaly_mask, 1],
            c="#005EB8", alpha=0.5, s=30, label="Normal")
plt.scatter(X_pca[anomaly_mask, 0],  X_pca[anomaly_mask, 1],
            c="#c62828", alpha=0.85, s=60, marker="X", label=f"Anomaly (top 5%)")
plt.title("KNN Anomaly Detection — PCA Projection", fontsize=13, fontweight="bold")
plt.xlabel(f"PC1 ({explained[0]:.1f}% variance)")
plt.ylabel(f"PC2 ({explained[1]:.1f}% variance)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# KNN distance distribution
plt.figure(figsize=(8, 4))
plt.hist(knn_dist, bins=40, color="#005EB8", alpha=0.75, edgecolor="white")
plt.axvline(threshold, color="#c62828", linewidth=2, linestyle="--",
            label=f"95th pct threshold ({threshold:.2f})")
plt.title("KNN Average Neighbour Distance Distribution", fontsize=12, fontweight="bold")
plt.xlabel("Avg Distance to 5 Nearest Neighbours")
plt.ylabel("Count")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# Anomaly profile
print("Anomaly patients — average feature values:")
anomaly_df = df_clean[anomaly_mask.tolist()][
    ["Age", "RestingBP", "Cholesterol", "MaxHR", "Oldpeak", "HeartDisease"]
].mean().round(2)
normal_df = df_clean[(~pd.Series(anomaly_mask)).tolist()][
    ["Age", "RestingBP", "Cholesterol", "MaxHR", "Oldpeak", "HeartDisease"]
].mean().round(2)

comparison = pd.DataFrame({"Anomaly": anomaly_df, "Normal": normal_df})
print(comparison.to_string())
print()
print("Done — all plots displayed.")
