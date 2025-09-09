import matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score

data = fetch_olivetti_faces(shuffle=True, random_state=42)
X = data.data 
y_true = data.target  

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
    return name, ari, nmi, sil

results = []
results.append(evaluate("K-Means", y_kmeans))
results.append(evaluate("Agglomerative", y_agg))
results.append(evaluate("DBSCAN", y_dbscan))

print("=== Comparação de Clusterização (Olivetti Faces) ===")
print(f"{'Algoritmo':15s} {'ARI':>8s} {'NMI':>8s} {'Silhouette':>12s}")
for name, ari, nmi, sil in results:
    print(f"{name:15s} {ari:8.4f} {nmi:8.4f} {sil if sil else 'NA':>12}")
