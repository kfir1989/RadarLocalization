from sklearn.cluster import DBSCAN
import numpy as np
from scipy.spatial.distance import cdist

def detectCarsInARow(X):
    clusters = DBSCAN(eps=1.5, min_samples=1).fit_predict(X)
    n_clusters = len(np.unique(clusters))-1
    max_dist = np.max(cdist(X,X))
    if n_clusters > 2 and max_dist/n_clusters < 4:
        return True
    else:
        return False
