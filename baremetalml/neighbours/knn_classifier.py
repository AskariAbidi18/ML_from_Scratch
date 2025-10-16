import numpy as np
from baremetalml import BaseModel
class KNNClassifier(BaseModel):
    def __init__(self, n_neighbors = 5, metric = 'euclidean', p = 2):
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.p = p

    def fit(self, X, y):
        X, y = self.check_x_y(X, y)
        self.X_train = X.astype(float)  
        self.y_train = y.astype(int)  
        X, y = self.check_x_y(X, y)
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        X = self.check_x(X).astype(float)
        predictions = []

        for x in X:
            if self.metric == 'euclidean':
                dist = np.sqrt(np.sum((self.X_train - x)**2, axis=1))
            elif self.metric == 'manhattan':
                dist = np.sum(np.abs(self.X_train - x), axis=1)
            elif self.metric == 'minkowski':
                dist = np.sum(np.abs(self.X_train - x)**self.p, axis=1)**(1/self.p)

            neighbors_idx = np.argsort(dist)[:self.n_neighbors]

            counts = np.bincount(self.y_train[neighbors_idx])
            pred = np.argmax(counts)
            predictions.append(pred)

        return np.array(predictions)
    