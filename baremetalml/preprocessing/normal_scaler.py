import numpy as np
from baremetalml.base.transformer import BaseTransformer
class NormalScaler(BaseTransformer):
    def fit(self, X):
        X = self.check_x(X)
        self.min = X.min(axis=0)
        self.max = X.max(axis=0)
        
    def transform(self, X):
        X = self.check_x(X)
        range_ = self.max - self.min
        range_[range_ == 0] = 1           # avoid division by zero
        X_norm = (X - self.min) / range_  # normalize
        return X_norm
    



        
        
