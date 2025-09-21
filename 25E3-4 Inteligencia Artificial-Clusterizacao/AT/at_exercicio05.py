import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage

url = "https://raw.githubusercontent.com/SteffiPeTaffy/machineLearningAZ/refs/heads/master/Machine%20Learning%20A-Z%20Template%20Folder/Part%204%20-%20Clustering/Section%2025%20-%20Hierarchical%20Clustering/Mall_Customers.csv"
df = pd.read_csv(url)

X = df[["Age", "Annual Income (k$)", "Spending Score (1-100)"]]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
linkages = ["ward", "average", "single", "complete"]

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for i, method in enumerate(linkages):
    Z = linkage(X_scaled, method=method)
    dendrogram(Z, truncate_mode="level", p=5, ax=axes[i])
    axes[i].set_title(f"Dendrograma - Linkage: {method}")
    axes[i].set_xlabel("Clientes")
    axes[i].set_ylabel("Distância")

plt.tight_layout()
plt.show()
