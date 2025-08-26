import matplotlib.pyplot as plt
from sklearn.datasets import make_swiss_roll
from scipy.spatial.distance import pdist, squareform
import seaborn as sns
import numpy as np

X, t = make_swiss_roll(n_samples=1000, noise=0.1, random_state=42)
np.random.seed(42)
idx = np.random.choice(len(X), size=100, replace=False)
X_sample = X[idx]
dist_matrix = squareform(pdist(X_sample, metric='euclidean'))

sns.clustermap(dist_matrix, method="ward", cmap="viridis", figsize=(10, 8))
plt.suptitle("Dendrograma + Mapa de Calor (Swiss Roll - amostra de 100 pontos)", y=1.02)
plt.show()
