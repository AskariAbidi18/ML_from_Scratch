import numpy as np 

class BaseTransformer:
    def fit(self, X):
        raise NotImplementedError("Subclass Must Implement The Function")
    
    def transform(self, X):
        raise NotImplementedError("Subclass Must Implement The Function")
    
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
    
    def check_x(self, X):
        if X.ndim != 2:
            raise ValueError("X must be 2D (samples x features)")
        return X
    