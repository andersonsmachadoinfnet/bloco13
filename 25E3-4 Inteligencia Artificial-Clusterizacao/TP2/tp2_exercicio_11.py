import matplotlib.pyplot as plt
from sklearn.datasets import make_swiss_roll
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import numpy as np

X, t = make_swiss_roll(n_samples=500, noise=0.1, random_state=42)
np.random.seed(42)
idx = np.random.choice(len(X), size=100, replace=False)
X_sample = X[idx]

Z = linkage(X_sample, method='ward')
distances = Z[:, 2]

# Calcular o salto entre distâncias consecutivas
diffs = np.diff(distances)
# Identificar o maior salto
max_gap_index = np.argmax(diffs)
optimal_clusters = len(X_sample) - (max_gap_index + 1)

print("Maior salto na distância entre clusters:", diffs[max_gap_index])
print("Número ótimo de clusters estimado:", optimal_clusters)

plt.figure(figsize=(12, 6))
dendrogram(Z, color_threshold=distances[max_gap_index])
plt.axhline(y=distances[max_gap_index], color='r', linestyle='--', label=f'Corte: {optimal_clusters} clusters')
plt.legend()
plt.title("Dendrograma com linha de corte (método do maior salto)")
plt.show()
