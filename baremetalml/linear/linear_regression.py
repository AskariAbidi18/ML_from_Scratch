import numpy as np
from baremetalml import BaseModel

class LinearRegression(BaseModel):
    def __init__(self, learning_rate=0.01, n_iterations=1000, method="gradient_descent", fit_intercept=True):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.method = method
        self.fit_intercept = fit_intercept
        self.weights = None
        self.bias = 0

    def fit(self, X, y):
        X, y = self.check_x_y(X, y)
        self.n_samples, self.n_features = X.shape

        if self.method == "normal_equation":
            ones = np.ones((X.shape[0], 1))        
            X_modified = np.hstack((ones, X))   
            self.weights = np.linalg.inv(X_modified.T @ X_modified) @ X_modified.T @ y  

            if self.fit_intercept:
                self.bias = self.weights[0]
                self.weights = self.weights[1:]
            else:
                self.bias = 0

        elif self.method == "gradient_descent":
            self.weights = np.zeros(self.n_features)
            self.bias = 0
            for _ in range(self.n_iterations):
                self.predictions = X @ self.weights + self.bias
                self.errors = self.predictions - y

                # Gradients calculation
                self.dw = (1/self.n_samples) * X.T @ self.errors
                self.db = (1/self.n_samples) * np.sum(self.errors)

                # Update weights and bias
                self.weights -= self.learning_rate * self.dw
                self.bias -= self.learning_rate * self.db

    def predict(self, X):
        X = self.check_x(X)
        return X @ self.weights + self.bias

    def compute_loss(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)
