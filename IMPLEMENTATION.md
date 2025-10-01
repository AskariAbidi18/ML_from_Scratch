# BareMetalML Implementation Guide

This document provides a **deep dive** into the **mathematical foundations** and **practical implementation** of BareMetalML components.  
It is intended for **learning, experimentation, and educational purposes**.

All classes are **modular**, so you can import them directly:

```python
from baremetalml import LinearRegression, StandardScaler, KNNClassifier
```

---

## 1. Base Classes

### 1.1 BaseModel

**Purpose**: Abstract class for all models with common interfaces and input validation.

**Responsibilities**:
- `fit(X, y)` – Train the model
- `predict(X)` – Make predictions
- **Input validation**: `check_x_y` and `check_x`

**Code Snippet**:

```python
class BaseModel:
    def fit(self, X, y):
        raise NotImplementedError
    def predict(self, X):
        raise NotImplementedError
```

**Why it matters**: Ensures consistency and reduces repetitive code across models.

---

### 1.2 BaseTransformer

**Purpose**: Abstract class for all data transformers.

**Methods**:
- `fit(X)` – Learn parameters from data
- `transform(X)` – Apply transformation
- `fit_transform(X)` – Combines fit + transform

**Code Snippet**:

```python
class BaseTransformer:
    def fit(self, X):
        raise NotImplementedError
    def transform(self, X):
        raise NotImplementedError
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
```

---

## 2. Linear Regression

### 2.1 Mathematical Formulation

**Linear regression predicts**:

$$\hat{y} = X\beta + \epsilon$$

*Where*:
- $X \in \mathbb{R}^{n \times d}$ = input matrix
- $\beta \in \mathbb{R}^{d}$ = weights
- $\epsilon$ = error

**Mean Squared Error (MSE)**:

$$\text{MSE} = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2$$

**Normal Equation (Analytical solution)**:

$$\hat{\beta} = (X^T X)^{-1} X^T y$$

**Gradient Descent (Iterative solution)**:

$$\beta := \beta - \alpha \frac{1}{n} X^T (X\beta - y)$$

Where $\alpha$ = learning rate.

---

### 2.2 Implementation in BareMetalML

```python
lr = LinearRegression(method="gradient_descent", learning_rate=0.01, n_iterations=1000)
lr.fit(X, y)
y_pred = lr.predict(X)
```

**Highlights**:
- Supports Normal Equation & Gradient Descent
- Automatically handles bias/intercept
- Computes predictions as: $\hat{y} = X \cdot \text{weights} + \text{bias}$

---

## 3. Logistic Regression

### 3.1 Mathematical Formulation

**Sigmoid function**:

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

**Prediction**:

$$\hat{y} = \sigma(X\beta)$$

**Binary Cross-Entropy Loss**:

$$L(\beta) = -\frac{1}{n}\sum_{i=1}^{n}\left[y_i \log(\hat{y}_i) + (1-y_i)\log(1-\hat{y}_i)\right]$$

**Gradient Descent Updates**:

$$\beta := \beta - \alpha \frac{1}{n} X^T (\hat{y} - y)$$

---

### 3.2 Implementation

```python
logr = LogisticRegression(n_iterations=1000, learning_rate=0.01)
logr.fit(X, y)
y_pred = logr.predict(X)
```

**Features**:
- Computes probabilities using sigmoid
- Updates weights via gradient of cross-entropy loss
- Predicts 0/1 based on 0.5 threshold

---

## 4. K-Nearest Neighbors (KNN)

### 4.1 Mathematical Formulation

**Distance Metrics**:

- **Euclidean**: $d = \sqrt{\sum(x_i - x_j)^2}$
- **Manhattan**: $d = \sum|x_i - x_j|$
- **Minkowski**: $d = \left(\sum|x_i - x_j|^p\right)^{1/p}$

**Prediction Rules**:
- **Classification**: majority vote of k nearest neighbors
- **Regression**: mean of k nearest neighbors

---

### 4.2 Implementation

```python
knn = KNNClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
```

**Pipeline Illustration**:

```
X_test → compute distances → select k nearest neighbors → predict majority class
```

---

## 5. Transformers

### 5.1 StandardScaler

**Equation**:

$$X_{\text{scaled}} = \frac{X - \mu}{\sigma}$$

---

### 5.2 NormalScaler

**Equation**:

$$X_{\text{norm}} = \frac{X - X_{\min}}{X_{\max} - X_{\min}}$$

---

### 5.3 LabelEncoder

Maps categorical labels to integers.

**Example**: `{'cat': 0, 'dog': 1, 'bird': 2}`

---

### 5.4 OneHotEncoder

Converts categories to one-hot vectors.

**Example**:

```
['red', 'blue'] → [[1, 0], [0, 1]]
```

---

### 5.5 PolynomialFeatures

Generates all polynomial combinations up to degree $d$:

$$(x_1, x_2) \rightarrow (1, x_1, x_2, x_1^2, x_1x_2, x_2^2)$$

```python
poly = PolynomialFeatures(degree=2, include_bias=True)
X_poly = poly.fit_transform(X)
```

---

## 6. Example Pipeline

```python
from baremetalml import StandardScaler, PolynomialFeatures, LinearRegression
import numpy as np

X = np.array([[1,2],[2,3],[3,4]])
y = np.array([3,5,7])

# Step 1: Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 2: Polynomial features
poly = PolynomialFeatures(degree=2, include_bias=True)
X_poly = poly.fit_transform(X_scaled)

# Step 3: Linear Regression
lr = LinearRegression(method='normal_equation')
lr.fit(X_poly, y)
y_pred = lr.predict(X_poly)
print("Predictions:", y_pred)
```

**Pipeline Overview**:

```
Raw Data → Scaling → Polynomial Feature Expansion → Linear Regression → Predictions
```

---

## Notes

- All models and transformers are pure NumPy, easy to inspect and extend
- Designed for learning, experimentation, and building pipelines from scratch
- Modular imports make it simple to use any component:

```python
from baremetalml import LinearRegression, StandardScaler, KNNClassifier
```