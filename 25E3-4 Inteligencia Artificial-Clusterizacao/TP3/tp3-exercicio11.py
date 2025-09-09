import matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score

data = fetch_olivetti_faces(shuffle=True, random_state=42)
X = data.data     # 400 amostras, 4096 atributos
y_true = data.target  # 40 classes (identidades)

pca = PCA(n_components=50, random_state=42)
X_pca = pca.fit_transform(X)
kmeans = KMeans(n_clusters=40, random_state=42, n_init=10)
y_kmeans = kmeans.fit_predict(X_pca)
agg = AgglomerativeClustering(n_clusters=40, linkage="ward")
y_agg = agg.fit_predict(X_pca)
dbscan = DBSCAN(eps=3, min_samples=5, n_jobs=-1)
y_dbscan = dbscan.fit_predict(X_pca)

def safe_silhouette(X, labels):
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    if n_clusters > 1:
        return silhouette_score(X, labels)
    else:
        return None

def evaluate(name, y_pred):
    ari = adjusted_rand_score(y_true, y_pred)
    nmi = normalized_mutual_info_score(y_true, y_pred)
    sil = safe_silhouette(X_pca, y_pred)
    print(f"{name:15s} -> ARI: {ari:.4f}, NMI: {nmi:.4f}, Silhouette: {sil if sil else 'NA'}")

print("=== Resultados de Clusterização (Olivetti Faces) ===")
evaluate("K-Means", y_kmeans)
evaluate("Agglomerative", y_agg)
evaluate("DBSCAN", y_dbscan)

pca_2d = PCA(n_components=2, random_state=42)
X_2d = pca_2d.fit_transform(X_pca)

fig, axes = plt.subplots(1, 4, figsize=(18, 5))

axes[0].scatter(X_2d[:, 0], X_2d[:, 1], c=y_true, cmap="tab20", s=15)
axes[0].set_title("Classes Reais")
axes[1].scatter(X_2d[:, 0], X_2d[:, 1], c=y_kmeans, cmap="tab20", s=15)
axes[1].set_title("K-Means")
axes[2].scatter(X_2d[:, 0], X_2d[:, 1], c=y_agg, cmap="tab20", s=15)
axes[2].set_title("Agglomerative")
axes[3].scatter(X_2d[:, 0], X_2d[:, 1], c=y_dbscan, cmap="tab20", s=15)
axes[3].set_title("DBSCAN")

plt.tight_layout()
plt.show()
