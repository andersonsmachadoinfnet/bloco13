import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

url = "https://raw.githubusercontent.com/SteffiPeTaffy/machineLearningAZ/refs/heads/master/Machine%20Learning%20A-Z%20Template%20Folder/Part%204%20-%20Clustering/Section%2025%20-%20Hierarchical%20Clustering/Mall_Customers.csv"
df = pd.read_csv(url)

X = df[["Age", "Annual Income (k$)", "Spending Score (1-100)"]]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# A) k-means
k = 5
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
labels = kmeans.fit_predict(X_scaled)

df["Cluster"] = labels

print("Centroides dos clusters (quantização vetorial):")
print(pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_), 
                   columns=X.columns))

plt.figure(figsize=(8,6))
plt.scatter(df["Annual Income (k$)"], df["Spending Score (1-100)"], 
            c=labels, cmap="viridis", s=50)
plt.scatter(scaler.inverse_transform(kmeans.cluster_centers_)[:,1], 
            scaler.inverse_transform(kmeans.cluster_centers_)[:,2], 
            c="red", marker="X", s=200, label="Centroides")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.title("Quantização Vetorial (K-Means)")
plt.legend()
plt.show()

# b PCA
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_scaled)

print("\nVariância explicada por cada componente principal:")
print(pca.explained_variance_ratio_)
print("\nComponentes principais (pesos das variáveis):")
print(pd.DataFrame(pca.components_, columns=X.columns, 
                   index=["PC1", "PC2", "PC3"]))

plt.figure(figsize=(8,6))
plt.scatter(X_pca[:,0], X_pca[:,1], c=labels, cmap="viridis", s=50)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Dados projetados em 2D via PCA (coloridos pelos clusters do K-Means)")
plt.show()

# c Comparação
print("\nComparação:")
print("- O K-means encontrou {} clusters (quantização vetorial dos clientes).".format(k))
print("- O PCA mostra que PC1 e PC2 explicam cerca de {:.2f}% da variância.".format(
    100*(pca.explained_variance_ratio_[0] + pca.explained_variance_ratio_[1])
))