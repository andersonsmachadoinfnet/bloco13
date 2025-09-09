import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score

X, y_true = make_circles(n_samples=500, factor=0.5, noise=0.05, random_state=42)

kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
y_kmeans = kmeans.fit_predict(X)
agg = AgglomerativeClustering(n_clusters=2, linkage="single")
y_agg = agg.fit_predict(X)
dbscan = DBSCAN(eps=0.1, min_samples=5)
y_dbscan = dbscan.fit_predict(X)

def safe_silhouette(X, labels):
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    if n_clusters > 1:
        return silhouette_score(X, labels)
    else:
        return None

sil_kmeans = safe_silhouette(X, y_kmeans)
sil_agg = safe_silhouette(X, y_agg)
sil_dbscan = safe_silhouette(X, y_dbscan)

print("=== Silhouette Coefficient ===")
print(f"K-Means        -> {sil_kmeans:.4f}")
print(f"Agglomerative  -> {sil_agg:.4f}")
print(f"DBSCAN         -> {sil_dbscan:.4f}" if sil_dbscan is not None else "DBSCAN -> Não foi possível calcular")

fig, axes = plt.subplots(1, 4, figsize=(18, 5))
axes[0].scatter(X[:, 0], X[:, 1], c=y_true, cmap="viridis", s=30, edgecolor="k")
axes[0].set_title("Classes Reais")
axes[1].scatter(X[:, 0], X[:, 1], c=y_kmeans, cmap="viridis", s=30, edgecolor="k")
axes[1].set_title(f"K-Means\nSil={sil_kmeans:.2f}")
axes[2].scatter(X[:, 0], X[:, 1], c=y_agg, cmap="viridis", s=30, edgecolor="k")
axes[2].set_title(f"Agglomerative\nSil={sil_agg:.2f}")
axes[3].scatter(X[:, 0], X[:, 1], c=y_dbscan, cmap="viridis", s=30, edgecolor="k")
axes[3].set_title(f"DBSCAN\nSil={sil_dbscan:.2f}" if sil_dbscan is not None else "DBSCAN\nSil=NA")

plt.tight_layout()
plt.show()
