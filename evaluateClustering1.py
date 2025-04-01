# -*- coding: utf-8 -*-
"""
Created on Mon Mar 31 09:07:18 2025

@author: USER
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    homogeneity_score,
    completeness_score,
    v_measure_score
)
from scipy.optimize import linear_sum_assignment

# 載入 iris 資料集，只取前兩個特徵
iris = datasets.load_iris()
X = iris.data[:, :2]
y_true = iris.target

# 執行 KMeans 分群
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
y_pred = kmeans.fit_predict(X)

# 自訂函數：計算 Clustering Accuracy（需重新對應 label）
def clustering_accuracy(y_true, y_pred):
    D = max(y_pred.max(), y_true.max()) + 1
    cost_matrix = np.zeros((D, D), dtype=int)
    for i in range(y_pred.size):
        cost_matrix[y_pred[i], y_true[i]] += 1
    row_ind, col_ind = linear_sum_assignment(cost_matrix.max() - cost_matrix)
    mapping = dict(zip(row_ind, col_ind))
    return sum(cost_matrix[i, j] for i, j in zip(row_ind, col_ind)) / y_pred.size, mapping

# 計算各種指標
ari = adjusted_rand_score(y_true, y_pred)
nmi = normalized_mutual_info_score(y_true, y_pred)
hom = homogeneity_score(y_true, y_pred)
comp = completeness_score(y_true, y_pred)
v_measure = v_measure_score(y_true, y_pred)
acc,mapping = clustering_accuracy(y_true, y_pred)

# 顯示指標
print("Adjusted Rand Index:", ari)
print("Normalized Mutual Info:", nmi)
print("Homogeneity:", hom)
print("Completeness:", comp)
print("V-measure:", v_measure)
print("Clustering Accuracy:", acc)

# 視覺化分群結果
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='viridis', edgecolor='k', s=60)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
            c='red', s=200, marker='X', label='Centers')
plt.title("KMeans Clustering on Iris (2D)")
plt.xlabel("Sepal length")
plt.ylabel("Sepal width")
plt.legend()
plt.grid(True)
plt.show()

# In[]- Clustering Acc
accuracy=0
import itertools

for perm in list(itertools.permutations(range(3))): #[(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]
    global y_pred
    print("y_pred:")    
    print(y_pred)
    labels = y_pred.copy()
    for i in range(len(y_pred)):
        labels[i] = perm[y_pred[i]]  # 重排標籤
    accuracy = max(accuracy, np.sum(labels == y_true) / len(y_pred))    
    print("Adjusted:")
    print(labels)
    print(y_true)
    print(f'Match Accuracy: {100 * accuracy :.2f}%')
    
    
print(f'Match Accuracy: {100 * accuracy :.2f}%')
