import numpy as np
from sklearn.datasets import make_circles
from sklearn.cluster import DBSCAN
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from scipy.stats import mode

X, y_true = make_circles(n_samples=500, factor=0.5, noise=0.05, random_state=42)

dbscan = DBSCAN(eps=0.1, min_samples=5)
y_pred = dbscan.fit_predict(X)

mask = y_pred != -1
y_pred_filtered = y_pred[mask]
y_true_filtered = y_true[mask]

labels_map = {}
for cluster in np.unique(y_pred_filtered):
    mask_cluster = y_pred_filtered == cluster
    majority_class = mode(y_true_filtered[mask_cluster], keepdims=False).mode
    labels_map[cluster] = majority_class

y_pred_mapped = np.array([labels_map[label] for label in y_pred_filtered])

acc = accuracy_score(y_true_filtered, y_pred_mapped)
prec = precision_score(y_true_filtered, y_pred_mapped)
rec = recall_score(y_true_filtered, y_pred_mapped)
f1 = f1_score(y_true_filtered, y_pred_mapped)

print(f"Acurácia: {acc:.4f}")
print(f"Precisão: {prec:.4f}")
print(f"Recall: {rec:.4f}")
print(f"F1-score: {f1:.4f}")
