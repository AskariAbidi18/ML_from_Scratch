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
        n_samples, n_features = X.shape

        if self.method == "normal_equation":
            X_mod = np.hstack((np.ones((n_samples, 1)), X)) if self.fit_intercept else X

            lambda_ = 1e-8
            theta = np.linalg.inv(X_mod.T @ X_mod + lambda_ * np.eye(X_mod.shape[1])) @ (X_mod.T @ y)

            if self.fit_intercept:
                self.bias = theta[0]
                self.weights = theta[1:]
            else:
                self.bias = 0
                self.weights = theta

        elif self.method == "gradient_descent":
            self.weights = np.zeros(n_features)
            self.bias = 0

            for _ in range(self.n_iterations):
                y_pred = X @ self.weights + self.bias
                errors = y_pred - y

                dw = (1 / n_samples) * (X.T @ errors)
                db = (1 / n_samples) * np.sum(errors)

                self.weights -= self.learning_rate * dw
                self.bias -= self.learning_rate * db

        else:
            raise ValueError(f"Unknown method: {self.method}")

    def predict(self, X):
        X = self.check_x(X)
        return X @ self.weights + self.bias

    def compute_loss(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)
