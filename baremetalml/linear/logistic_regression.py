import numpy as np
from baremetalml import BaseModel
class LogisticRegression(BaseModel):
    def __init__(self, learning_rate = 0.01, n_iterations = 1000, fit_intercept = True):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.fit_intercept = fit_intercept
        self.weights = None
        self.bias = 0
        
    
    def fit(self, X, y):
        X, y = self.check_x_y(X, y)
        n_samples, n_features = X.shape

        self.weights = np.zeros(n_features)
        self.bias = 0

        for i in range(self.n_iterations):
            linear_output = X @ self.weights + self.bias
            y_prediction = 1 / (1+ np.exp(-linear_output))

            error = y_prediction - y

            #gradient calculation
            dw = (1 / n_samples) * (X.T @ error)
            db = (1 / n_samples) * sum(error)

            #update weights 
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict_proba(self, X):
        X = self.check_x(X)
        linear_output = X @ self.weights + self.bias
        return 1 / (1 + np.exp(-linear_output))

    def predict(self, X):
        probs = self.predict_proba(X)
        return (probs >= 0.5).astype(int)
        

    def compute_loss(self, y_true, y_pred):
        return -np.mean(y_true * np.log(y_pred + 1e-15) + (1 - y_true) * np.log(1 - y_pred + 1e-15))
