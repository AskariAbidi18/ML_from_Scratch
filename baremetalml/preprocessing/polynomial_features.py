import numpy as np
from itertools import combinations_with_replacement
from baremetalml.base.transformer import BaseTransformer
class PolynomialFeatures(BaseTransformer):
    def __init__(self, degree = 2, include_bias = True):
        self.degree = degree
        self.include_bias = include_bias 

    def fit(self, X):
        X = self.check_x(X)
        return self 

    def transform(self, X):
        X = self.check_x(X)
        n_samples, n_features = X.shape
        output = list()

        for sample in X:
            row_features = list()
                    
            if self.include_bias:
                row_features.append(1)

            for d in range(1, self.degree + 1):
                for combo in combinations_with_replacement(range(n_features), d):
                    prod = 1
                    for index in combo:
                        prod *= sample[index]
                        
                    row_features.append(prod)
                
            output.append(row_features)
        return np.array(output)
