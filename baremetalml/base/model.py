import numpy as np
class BaseModel :
    def fit(self, X, y):
        raise NotImplementedError("Subclass Must Implement The Function")
    
    def predict(self, X):
        raise NotImplementedError("Subclass Must Implement The Function")
    
    def check_x_y(self, X, y):
        X = np.array(X)
        y = np.array(y)
        if X.shape != y.shape:
            raise ValueError(f"X and y have incompatible shapes X : {X.shape()} vs y : {y.shape()}")
        return X, y

    def check_x(self, X):
        X = np.array(X)
        return X
    
    