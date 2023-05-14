import numpy as np

class KNearestNeighbor(object):
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
    
    def _hellinger(self, X, S):
        # S is an N x D matrix where each row is an example
        # X is a single D-dimensional example
        
        return 1 - np.sum(np.sqrt(X * S), axis=1)

    def _histogram_intersection(self, X, S):
        return 1 - np.sum(np.minimum(X, S), axis=1)

    def predict(self, X, method):
        # compute distances between X and all examples in the training set
        if method == "histogram_intersection":
            dists = self._histogram_intersection(X, self.X_train)
        elif method == "hellinger":
            dists = self._hellinger(X, self.X_train)
        else:
            raise ValueError(f"Invalid method: {method}")

        # Return the indices of k neighbors
        idx = np.argsort(dists)[:self.k]

        # Extract the labels of the k nearest neighbor training samples
        k_closest_classes = self.y_train[idx]

        # return the most common class label, the indices of the closest neighbors, and the distances to the closest neighbors
        return (np.argmax(np.bincount(k_closest_classes)), idx, dists[idx])