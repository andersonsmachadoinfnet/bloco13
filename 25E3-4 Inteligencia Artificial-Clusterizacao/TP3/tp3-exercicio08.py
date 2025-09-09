import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

X, y_true = make_circles(n_samples=500, factor=0.5, noise=0.05, random_state=42)

kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
y_kmeans = kmeans.fit_predict(X)
agg = AgglomerativeClustering(n_clusters=2, linkage="single")
y_agg = agg.fit_predict(X)
dbscan = DBSCAN(eps=0.1, min_samples=5)
y_dbscan = dbscan.fit_predict(X)

def get_scores(y_true, y_pred):
    ari = adjusted_rand_score(y_true, y_pred)
    nmi = normalized_mutual_info_score(y_true, y_pred)
    return ari, nmi

ari_kmeans, nmi_kmeans = get_scores(y_true, y_kmeans)
ari_agg, nmi_agg = get_scores(y_true, y_agg)
ari_dbscan, nmi_dbscan = get_scores(y_true, y_dbscan)

print("=== Resultados ===")
print(f"K-Means        -> ARI: {ari_kmeans:.4f}, NMI: {nmi_kmeans:.4f}")
print(f"Agglomerative  -> ARI: {ari_agg:.4f}, NMI: {nmi_agg:.4f}")
print(f"DBSCAN         -> ARI: {ari_dbscan:.4f}, NMI: {nmi_dbscan:.4f}")

fig, axes = plt.subplots(1, 4, figsize=(18, 5))
axes[0].scatter(X[:, 0], X[:, 1], c=y_true, cmap="viridis", s=30, edgecolor="k")
axes[0].set_title("Classes Reais")
axes[1].scatter(X[:, 0], X[:, 1], c=y_kmeans, cmap="viridis", s=30, edgecolor="k")
axes[1].set_title(f"K-Means\nARI={ari_kmeans:.2f}, NMI={nmi_kmeans:.2f}")
axes[2].scatter(X[:, 0], X[:, 1], c=y_agg, cmap="viridis", s=30, edgecolor="k")
axes[2].set_title(f"Agglomerative\nARI={ari_agg:.2f}, NMI={nmi_agg:.2f}")
axes[3].scatter(X[:, 0], X[:, 1], c=y_dbscan, cmap="viridis", s=30, edgecolor="k")
axes[3].set_title(f"DBSCAN\nARI={ari_dbscan:.2f}, NMI={nmi_dbscan:.2f}")

plt.tight_layout()
plt.show()
