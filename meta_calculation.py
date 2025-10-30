import numpy as np
from itertools import combinations
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

def get_class_separability(X, y):
    X = pd.get_dummies(X)
    X = np.array(X)
    y = np.array(y)

    unique_classes = np.unique(y)

    centroids = {}
    for cls in unique_classes:
        centroids[cls] = np.mean(X[y == cls], axis=0)

    distances = []
    for c1, c2 in combinations(unique_classes, 2):
        # d = np.sqrt(np.sum((centroids[c1] - centroids[c2]) ** 2))
        d = np.linalg.norm(centroids[c1] - centroids[c2])
        distances.append(d)
    
    mean_distance = np.mean(distances)

    class_sep = 0.5 + 1.5 * (mean_distance/(mean_distance + 5))

    return class_sep

def get_n_informative(X, y, threshold=0.01):
    X = pd.get_dummies(X)
    mi = mutual_info_classif(X, y, discrete_features='auto')
    n_info = np.sum(mi > threshold)
    return n_info