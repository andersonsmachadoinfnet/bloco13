import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.metrics import adjusted_rand_score, silhouette_score
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_covtype
data = fetch_covtype()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target  

X_sample = X.sample(n=5000, random_state=42)
y_sample = y[X_sample.index]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_sample)

dbscan = DBSCAN(eps=1.5, min_samples=10, n_jobs=-1)
labels = dbscan.fit_predict(X_scaled)

n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
print(f"Número de clusters: {n_clusters}")

if n_clusters > 1:
    ari = adjusted_rand_score(y_sample, labels)
    sil = silhouette_score(X_scaled, labels)
    print(f"Silhouette Score: {sil:.4f}")
else:
    print("DBSCAN não encontrou múltiplos clusters.")

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap="tab10", s=10, alpha=0.6)
plt.title("DBSCAN no dataset Forest CoverTypes")
plt.xlabel("Componente Principal 1")
plt.ylabel("Componente Principal 2")
plt.show()
