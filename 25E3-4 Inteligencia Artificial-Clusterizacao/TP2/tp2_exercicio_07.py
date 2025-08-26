import matplotlib.pyplot as plt
from sklearn.datasets import make_swiss_roll
from sklearn.cluster import AgglomerativeClustering
from mpl_toolkits.mplot3d import Axes3D

X, t = make_swiss_roll(n_samples=1000, noise=0.1, random_state=42)

# Aplicar clusterização hierárquica
hc = AgglomerativeClustering(n_clusters=6, linkage='ward')
labels = hc.fit_predict(X)

fig = plt.figure(figsize=(12, 5))
ax2 = fig.add_subplot(122)
ax2.scatter(X[:, 0], X[:, 2], c=labels, cmap='tab10', s=10)
ax2.set_title("Clusterização Hierárquica no Swiss Roll")

plt.show()
