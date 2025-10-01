BareMetalML Implementation Guide
This document provides a deep dive into the mathematical foundations and practical implementation of BareMetalML components.
It is intended for learning, experimentation, and educational purposes.

All classes are modular, so you can import them directly:

Python

from baremetalml import LinearRegression, StandardScaler, KNNClassifier
1. Base Classes
1.1 BaseModel
Purpose: Abstract class for all models with common interfaces and input validation.

Responsibilities:

fit(X, y) – Train the model

predict(X) – Make predictions

Input validation: check_x_y and check_x

Code Snippet:

Python

class BaseModel:
    def fit(self, X, y):
        raise NotImplementedError
    def predict(self, X):
        raise NotImplementedError
Why it matters: Ensures consistency and reduces repetitive code across models.

1.2 BaseTransformer
Purpose: Abstract class for all data transformers.

Methods:

fit(X) – Learn parameters from data

transform(X) – Apply transformation

fit_transform(X) – Combines fit + transform

Code Snippet:

Python

class BaseTransformer:
    def fit(self, X):
        raise NotImplementedError
    def transform(self, X):
        raise NotImplementedError
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
2. Linear Regression
2.1 Mathematical Formulation
Linear regression predicts the output  
y
^
​
  as:

$$\hat{y} = X\beta + \epsilon $$*Where*: * $X \in \mathbb{R}^{n \times d}$ = input matrix * $\beta \in \mathbb{R}^{d}$ = weights * $\epsilon$ = error **Mean Squared Error (MSE)**: The cost function to minimize. $$MSE = \frac{1}{n} \sum\_{i=1}^{n} (y\_i - \hat{y}\_i)^2 $$**Normal Equation (Analytical solution):** A direct formula to find the optimal weights. $$
\hat{\beta} = (X^T X)^{-1} X^T y
$$Gradient Descent (Iterative solution): An iterative approach to find the optimal weights.

$$\beta := \beta - \alpha \frac{1}{n} X^T (X\beta - y)
$$Where α = learning rate.

2.2 Implementation in BareMetalML
Python

from baremetalml import LinearRegression

lr = LinearRegression(method="gradient_descent", learning_rate=0.01, n_iterations=1000)
lr.fit(X, y)
y_pred = lr.predict(X)
Highlights:

Supports both Normal Equation & Gradient Descent.

Automatically handles the bias/intercept term.

Computes predictions as:  
y
^
​
 =X⋅weights+bias

3. Logistic Regression
3.1 Mathematical Formulation
Sigmoid function: Maps any real value into the range (0, 1), representing a probability.

$$\sigma(z) = \frac{1}{1 + e^{-z}} $$**Prediction**: The predicted probability is calculated by passing the linear model's output through the sigmoid function. $$\hat{y} = \sigma(X\beta) $$**Binary Cross-Entropy Loss**: The cost function for binary classification. $$
L(\beta) = -\frac{1}{n} \sum_{i=1}^{n} [y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i)]
$$Gradient Descent Updates: The weights are updated by taking steps in the direction opposite to the gradient of the loss function.

$$\beta := \beta - \alpha \frac{1}{n} X^T (\hat{y} - y)
$$### 3.2 Implementation

Python

from baremetalml import LogisticRegression

logr = LogisticRegression(n_iterations=1000, learning_rate=0.01)
logr.fit(X, y)
y_pred = logr.predict(X)
Computes probabilities using the sigmoid function.

Updates weights via the gradient of the cross-entropy loss.

Predicts class 0 or 1 based on a 0.5 probability threshold.

4. K-Nearest Neighbors (KNN)
4.1 Mathematical Formulation
Distance Metrics:

Euclidean: d= 
∑(x 
i
​
 −x 
j
​
 ) 
2
 

​
 

Manhattan: d=∑∣x 
i
​
 −x 
j
​
 ∣

Minkowski: d=(∑∣x 
i
​
 −x 
j
​
 ∣ 
p
 ) 
1/p
 

Prediction Rules:

Classification: Majority vote of the k nearest neighbors.

Regression: Mean of the k nearest neighbors.

4.2 Implementation
Python

from baremetalml import KNNClassifier

knn = KNNClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
Pipeline Illustration:

X_test -> Compute distances -> Select k nearest neighbors -> Predict majority class
5. Transformers
5.1 StandardScaler
Standardizes features by removing the mean and scaling to unit variance.

Equation:

$$X_{scaled} = \frac{X - \mu}{\sigma} $$### 5.2 NormalScaler Scales features to a given range, typically [0, 1]. **Equation**: $$X\_{norm} = \frac{X - X\_{min}}{X\_{max} - X\_{min}} $$\#\#\# 5.3 LabelEncoder Maps categorical labels to integer values. *Example*: `{'cat':0, 'dog':1, 'bird':2}` ### 5.4 OneHotEncoder Converts categorical integer features into one-hot encoded vectors. *Example*: $$
['red', 'blue'] \rightarrow \begin{bmatrix} 1 & 0 \ 0 & 1 \end{bmatrix}
$$### 5.5 PolynomialFeatures

Generates polynomial and interaction features. For degree d=2:

$$(x_1, x_2) \rightarrow (1, x_1, x_2, x_1^2, x_1 x_2, x_2^2)

$$ ```python from baremetalml import PolynomialFeatures poly = PolynomialFeatures(degree=2, include_bias=True) X_poly = poly.fit_transform(X) ``` ## 6\. Example Pipeline ```python from baremetalml import StandardScaler, PolynomialFeatures, LinearRegression import numpy as np # Sample Data X = np.array([[1,2],[2,3],[3,4]]) y = np.array([3,5,7]) # Step 1: Standardize features scaler = StandardScaler() X_scaled = scaler.fit_transform(X) # Step 2: Generate polynomial features poly = PolynomialFeatures(degree=2, include_bias=True) X_poly = poly.fit_transform(X_scaled) # Step 3: Train a Linear Regression model lr = LinearRegression(method='normal_equation') lr.fit(X_poly, y) # Step 4: Make predictions y_pred = lr.predict(X_poly) print("Predictions:", y_pred) ``` **Pipeline Overview**: ``` Raw Data -> Scaling -> Polynomial Feature Expansion -> Linear Regression -> Predictions ``` ## Notes * All models and transformers are built with pure **NumPy**, making them easy to inspect and extend. * This library is designed for **learning, experimentation, and building pipelines from scratch**. * Modular imports make it simple to use any component.$$