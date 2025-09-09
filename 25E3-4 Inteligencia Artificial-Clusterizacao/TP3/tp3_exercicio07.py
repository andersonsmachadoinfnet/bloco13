import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from sklearn.cluster import KMeans, AgglomerativeClustering

X, y_true = make_circles(n_samples=500, factor=0.5, noise=0.05, random_state=42)

kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
y_kmeans = kmeans.fit_predict(X)
agg = AgglomerativeClustering(n_clusters=2, linkage="single")
y_agg = agg.fit_predict(X)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].scatter(X[:, 0], X[:, 1], c=y_true, cmap="viridis", s=30, edgecolor="k")
axes[0].set_title("Classes Reais (make_circles)")
axes[1].scatter(X[:, 0], X[:, 1], c=y_kmeans, cmap="viridis", s=30, edgecolor="k")
axes[1].set_title("Clusterização com K-Means")
axes[2].scatter(X[:, 0], X[:, 1], c=y_agg, cmap="viridis", s=30, edgecolor="k")
axes[2].set_title("Clusterização com Agglomerative")

plt.show()
