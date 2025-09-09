import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from sklearn.cluster import DBSCAN
from sklearn.metrics import adjusted_rand_score

X, y_true = make_circles(n_samples=500, factor=0.5, noise=0.05, random_state=42)

eps_values = [0.05, 0.1, 0.15, 0.2, 0.3]
min_samples_values = [3, 5, 10]
results = []

for eps in eps_values:
    for min_samples in min_samples_values:
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        y_pred = dbscan.fit_predict(X)
        
        n_clusters = len(set(y_pred)) - (1 if -1 in y_pred else 0)
        ari = adjusted_rand_score(y_true, y_pred)
        results.append((eps, min_samples, n_clusters, ari))

print("eps\tmin_samples\tclusters\tARI")
for eps, min_samples, n_clusters, ari in results:
    print(f"{eps:.2f}\t{min_samples}\t\t{n_clusters}")

fig, axes = plt.subplots(len(eps_values), len(min_samples_values), figsize=(12, 10))

for i, eps in enumerate(eps_values):
    for j, min_samples in enumerate(min_samples_values):
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        y_pred = dbscan.fit_predict(X)
        
        axes[i, j].scatter(X[:, 0], X[:, 1], c=y_pred, cmap="viridis", s=20, edgecolor="k")
        axes[i, j].set_title(f"eps={eps}, min_samples={min_samples}")
        axes[i, j].set_xticks([])
        axes[i, j].set_yticks([])

plt.tight_layout()
plt.show()
