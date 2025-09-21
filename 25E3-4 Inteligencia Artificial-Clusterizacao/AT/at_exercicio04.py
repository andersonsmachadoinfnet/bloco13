import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

# 1. Carregar dataset
url = "https://raw.githubusercontent.com/SteffiPeTaffy/machineLearningAZ/refs/heads/master/Machine%20Learning%20A-Z%20Template%20Folder/Part%204%20-%20Clustering/Section%2025%20-%20Hierarchical%20Clustering/Mall_Customers.csv"
df = pd.read_csv(url)

# Selecionar variáveis numéricas
X = df[["Age", "Annual Income (k$)", "Spending Score (1-100)"]]

# Padronizar
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 2. Testar diferentes métodos de linkage
linkages = ["ward", "average", "single", "complete"]
n_clusters = 5  # número fixo para comparar

results = {}

for link in linkages:
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage=link)
    labels = model.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    results[link] = {
        "labels": labels,
        "silhouette": score
    }

# 3. Mostrar resultados
print("Resultados por linkage:")
for link, res in results.items():
    print(f"- {link.capitalize()}: Silhouette = {res['silhouette']:.4f}")

# 4. Visualização 2D (renda x gasto) para cada linkage
fig, axes = plt.subplots(2, 2, figsize=(12,10))
axes = axes.flatten()

for i, link in enumerate(linkages):
    labels = results[link]["labels"]
    axes[i].scatter(X["Annual Income (k$)"], X["Spending Score (1-100)"], 
                    c=labels, cmap="viridis", s=50)
    axes[i].set_title(f"Linkage: {link} (Silhouette={results[link]['silhouette']:.2f})")
    axes[i].set_xlabel("Annual Income (k$)")
    axes[i].set_ylabel("Spending Score")

plt.tight_layout()
plt.show()
