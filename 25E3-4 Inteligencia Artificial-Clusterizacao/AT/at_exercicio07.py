import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_swiss_roll
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import silhouette_score

n_samples = 1500
X, t = make_swiss_roll(n_samples, noise=0.05)
X_scaled = StandardScaler().fit_transform(X)

# Diferentes valores para eps e min_samples
eps_values = [2, 3, 4, 5]
min_samples_values = [5, 10, 15]
results = []

for eps in eps_values:
    for min_s in min_samples_values:
        db = DBSCAN(eps=eps, min_samples=min_s).fit(X)
        labels = db.labels_
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        if n_clusters > 1:
            score = silhouette_score(X, labels)
        else:
            score = -1
        results.append((eps, min_s, n_clusters, score))

# Mostrar resultados
print("eps | min_samples | n_clusters | silhouette")
for r in results:
    print(r)

dbscan = DBSCAN(eps=0.5, min_samples=10)
labels = dbscan.fit_predict(X_scaled)
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

scatter = ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=labels, cmap='vanimo', s=10)
ax.set_title("DBSCAN Clustering no Swiss Roll")
plt.colorbar(scatter, label="Cluster")
plt.show()
