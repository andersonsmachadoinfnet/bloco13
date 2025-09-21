import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_olivetti_faces
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score, silhouette_score

sns.set(style="whitegrid")
faces = fetch_olivetti_faces(shuffle=True, random_state=42)
X = faces.data
y = faces.target 

print("Formato do dataset:", X.shape)
print("Número de classes reais:", len(set(y)))

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
results = {}

# KMeans
kmeans = KMeans(n_clusters=40, random_state=42)
kmeans_labels = kmeans.fit_predict(X)
results["KMeans"] = {
    "labels": kmeans_labels,
    "ARI": adjusted_rand_score(y, kmeans_labels),
    "Silhouette": silhouette_score(X, kmeans_labels)
}

# DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=5)
dbscan_labels = dbscan.fit_predict(X)
results["DBSCAN"] = {
    "labels": dbscan_labels,
    "ARI": adjusted_rand_score(y, dbscan_labels),
    "Silhouette": silhouette_score(X, dbscan_labels) if len(set(dbscan_labels)) > 1 else -1
}

# Agglomerative Clustering
agg = AgglomerativeClustering(n_clusters=40)
agg_labels = agg.fit_predict(X)
results["Agglomerative"] = {
    "labels": agg_labels,
    "ARI": adjusted_rand_score(y, agg_labels),
    "Silhouette": silhouette_score(X, agg_labels)
}

# Resultados
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for i, (name, res) in enumerate(results.items()):
    sns.scatterplot(
        x=X_pca[:, 0], y=X_pca[:, 1],
        hue=res["labels"],
        palette="tab20", legend=None, ax=axes[i], s=30
    )
    axes[i].set_title(f"{name}\nARI: {res['ARI']:.2f}, Silhouette: {res['Silhouette']:.2f}")
    axes[i].set_xlabel("PCA 1")
    axes[i].set_ylabel("PCA 2")
plt.tight_layout()
plt.show()
