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
            # Add bias column only if fit_intercept is True
            if self.fit_intercept:
                ones = np.ones((X.shape[0], 1))
                X_modified = np.hstack((ones, X))
            else:
                X_modified = X

            # Use pseudo-inverse instead of inv
            self.weights = np.linalg.pinv(X_modified.T @ X_modified) @ X_modified.T @ y

            if self.fit_intercept:
                self.bias = self.weights[0]
                self.weights = self.weights[1:]
            else:
                self.bias = 0

        elif self.method == "gradient_descent":
            self.weights = np.zeros(self.n_features)
            self.bias = 0
            for _ in range(self.n_iterations):
                predictions = X @ self.weights + self.bias
                errors = predictions - y

                # gradients
                dw = (1 / self.n_samples) * X.T @ errors
                db = (1 / self.n_samples) * np.sum(errors)

                self.weights -= self.learning_rate * dw
                self.bias -= self.learning_rate * db

    def predict(self, X):
        X = self.check_x(X)
        return X @ self.weights + self.bias

    def compute_loss(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)
