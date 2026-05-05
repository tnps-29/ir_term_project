import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class RocchioClassifier:
    """
    Custom Nearest Centroid implementation strictly using Cosine Similarity
    to accurately implement the Rocchio classification algorithm.
    """
    def fit(self, X, y):
        y_array = np.array(y)
        self.classes_ = np.unique(y_array)
        self.centroids_ = []
        for c in self.classes_:
            mask = (y_array == c)
            centroid = X[mask].mean(axis=0)
            self.centroids_.append(np.asarray(centroid).flatten())
        self.centroids_ = np.array(self.centroids_)
        return self
        
    def predict(self, X):
        sims = cosine_similarity(X, self.centroids_)
        return self.classes_[np.argmax(sims, axis=1)]
        
    def predict_with_score(self, X):
        sims = cosine_similarity(X, self.centroids_)
        best_idx = np.argmax(sims, axis=1)
        return self.classes_[best_idx], np.max(sims, axis=1)
