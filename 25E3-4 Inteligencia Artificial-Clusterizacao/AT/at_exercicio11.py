import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score

sns.set(style="whitegrid")
newsgroups = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
texts = newsgroups.data
y_true = newsgroups.target
target_names = newsgroups.target_names
n_clusters = len(target_names)

tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.5, max_features=10000)
X_tfidf = tfidf_vectorizer.fit_transform(texts)
terms = tfidf_vectorizer.get_feature_names_out()

# K-MEANS
svd = TruncatedSVD(n_components=100, random_state=42)
X_reduced = svd.fit_transform(X_tfidf)
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
y_pred_kmeans = kmeans.fit_predict(X_reduced)
ari = adjusted_rand_score(y_true, y_pred_kmeans)
silhouette = silhouette_score(X_reduced, y_pred_kmeans)
print(f"K-Means - ARI: {ari:.3f}, Silhouette: {silhouette:.3f}")

# Visualização
pca = PCA(n_components=2, random_state=42)
X_2d = pca.fit_transform(X_reduced)
plt.figure(figsize=(10, 6))
sns.scatterplot(x=X_2d[:, 0], y=X_2d[:, 1], hue=y_pred_kmeans, palette='tab20', legend=None, s=10)
plt.title(f'K-Means Clustering (2D) | ARI: {ari:.2f}, Silhouette: {silhouette:.2f}')
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.show()

# NMF
print("\nDeterminando tópicos com NMF...")
nmf_model = NMF(n_components=10, random_state=42)
nmf_topics = nmf_model.fit_transform(X_tfidf)

def show_topics(model, feature_names, n_top_words=10):
    for idx, topic in enumerate(model.components_):
        top_terms = [feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]
        print(f"Tópico #{idx + 1}: {', '.join(top_terms)}")

print("\nTópicos NMF:")
show_topics(nmf_model, terms)

# LDA
count_vectorizer = CountVectorizer(stop_words='english', max_df=0.5, max_features=10000)
X_count = count_vectorizer.fit_transform(texts)
count_terms = count_vectorizer.get_feature_names_out()

lda_model = LatentDirichletAllocation(n_components=10, random_state=42, learning_method='batch')
lda_topics = lda_model.fit_transform(X_count)

print("\nTópicos LDA:")
show_topics(lda_model, count_terms)
