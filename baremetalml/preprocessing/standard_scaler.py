import numpy as np
from baremetalml import BaseTransformer
class StandardScaler(BaseTransformer):
    def fit(self, X):
        X = self.check_x(X)
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0)
        self.std[self.std == 0] = 1 
        
    def transform(self, X):
        X = self.check_x(X)       
        X_std = (X-self.mean) / self.std  
        return X_std
    


        
        
