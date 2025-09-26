import numpy as np
from itertools import combinations_with_replacement
from baremetalml import BaseTransformer

class PolynomialFeatures(BaseTransformer):
    def __init__(self, degree=2, include_bias=True):
        self.degree = degree
        self.include_bias = include_bias

    def fit(self, X):
        X = self.check_x(X)
        return self

    def transform(self, X):
        X = np.array(X, dtype=float)  # ensure numeric
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        n_samples, n_features = X.shape
        output = []

        for sample in X:
            row_features = []

            if self.include_bias:
                row_features.append(1)

            for d in range(1, self.degree + 1):
                for combo in combinations_with_replacement(range(n_features), d):
                    prod = 1
                    for index in combo:
                        prod *= sample[index]
                    row_features.append(prod)

            output.append(row_features)

        return np.array(output, dtype=float)
