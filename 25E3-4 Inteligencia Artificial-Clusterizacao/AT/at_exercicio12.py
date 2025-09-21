import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_openml
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X = mnist.data
y = mnist.target.astype(int)

print(f"Formato do dataset: {X.shape}")
print(f"Número de classes: {len(set(y))}")

from sklearn.utils import shuffle
X, y = shuffle(X, y, random_state=42)
X_sample = X[:3000]
y_sample = y[:3000]

X_sample = StandardScaler().fit_transform(X_sample)

# t-SNE
tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
X_tsne = tsne.fit_transform(X_sample)

# Resultado
sns.set(style="whitegrid", context="notebook")
plt.figure(figsize=(10, 7))
scatter = sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=y_sample, palette="tab10", legend="full", s=40)
plt.title("Visualização do MNIST com t-SNE (3000 amostras)")
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.legend(title="Dígitos", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()
