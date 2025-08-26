import matplotlib.pyplot as plt
from sklearn.datasets import make_swiss_roll
from scipy.cluster.hierarchy import dendrogram, linkage
import numpy as np

X, t = make_swiss_roll(n_samples=1000, noise=0.1, random_state=42)
np.random.seed(42)
idx = np.random.choice(len(X), size=100, replace=False)
X_sample = X[idx]
Z = linkage(X_sample, method='ward')

plt.figure(figsize=(12, 6))
dendrogram(Z, leaf_rotation=90, leaf_font_size=8)
plt.title("Dendrograma - Clusterização Hierárquica (amostra de 100 pontos do Swiss Roll)")
plt.xlabel("Amostras")
plt.ylabel("Distância Euclidiana")
plt.show()
