# BareMetalML Usage Guide

This guide shows how to use **BareMetalML** models, transformers, and utilities with simple code examples.  
All imports are **modular**, so you don’t need to know the internal file structure. For example:

```python
from baremetalml import LinearRegression, StandardScaler, KNNClassifier
```

## 1. Preprocessing and Transformers 

### StandardScaler 

Standardizes features to mean=0 and standard deviation=1:
```
import numpy as np
from baremetalml import StandardScaler

X = np.array([[1, 2], [3, 4], [5, 6]])
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print(X_scaled)
```

### NormalScaler

Scales features to the range [0,1]:
```
from baremetalml import NormalScaler

scaler = NormalScaler()
X_norm = scaler.fit_transform(X)
print(X_norm)
```

### LabelEncoder

Converts categorical labels to integers:
```
from baremetalml import LabelEncoder

y = ['cat', 'dog', 'cat', 'bird']
encoder = LabelEncoder()
y_encoded = encoder.fit(y)
y_transformed = encoder.transform(y)
print(y_transformed)
```

### OneHotEncoder

Converts categorical features into one-hot encoded vectors:
```
from baremetalml import OneHotEncoder
import numpy as np

X_cat = np.array([['red', 'S'], ['blue', 'M'], ['red', 'L']])
ohe = OneHotEncoder()
ohe.fit(X_cat)
X_encoded = ohe.transform(X_cat)
print(X_encoded)
```

### PolynomialFeatures

Generates polynomial and interaction features:
```
from baremetalml import PolynomialFeatures

X = np.array([[1, 2], [3, 4]])
poly = PolynomialFeatures(degree=2, include_bias=True)
X_poly = poly.fit_transform(X)
print(X_poly)
```

## 2. Models

### Linear Regression

Supports both Gradient Descent and Normal Equation:
```
from baremetalml import LinearRegression
import numpy as np

X = np.array([[1], [2], [3]])
y = np.array([2, 4, 6])

lr = LinearRegression(method='gradient_descent', learning_rate=0.1, n_iterations=1000)
lr.fit(X, y)
y_pred = lr.predict(X)
print(y_pred)
```

### Logistic Regression

Binary classification:
```
from baremetalml import LogisticRegression

X = np.array([[0], [1], [2], [3]])
y = np.array([0, 0, 1, 1])

logr = LogisticRegression(n_iterations=1000, learning_rate=0.1)
logr.fit(X, y)
y_pred = logr.predict(X)
print(y_pred)
```

### KNN Classifier

Classification with multiple distance metrics:
```
from baremetalml import KNNClassifier

X_train = np.array([[1], [2], [3]])
y_train = np.array([0, 0, 1])

knn = KNNClassifier(n_neighbors=2)
knn.fit(X_train, y_train)
y_pred = knn.predict(np.array([[1.5], [2.5]]))
print(y_pred)
```

### KNN Regressor

Regression using nearest neighbors:
```
from baremetalml import KNNRegressor

X_train = np.array([[1], [2], [3]])
y_train = np.array([2, 4, 6])

knn_reg = KNNRegressor(n_neighbors=2)
knn_reg.fit(X_train, y_train)
y_pred = knn_reg.predict(np.array([[1.5], [2.5]]))
print(y_pred)
```

## 3. Full ML Pipeline Example

Here’s a full workflow combining preprocessing, feature engineering, and modeling:
```
import numpy as np
from baremetalml import StandardScaler, PolynomialFeatures, LinearRegression

# Sample dataset
X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
y = np.array([3, 5, 7, 9])

# 1. Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 2. Generate polynomial features
poly = PolynomialFeatures(degree=2, include_bias=True)
X_poly = poly.fit_transform(X_scaled)

# 3. Train a Linear Regression model
lr = LinearRegression(method='normal_equation')
lr.fit(X_poly, y)

# 4. Make predictions
y_pred = lr.predict(X_poly)
print("Predictions:", y_pred)
```

This shows a complete pipeline:

Scaling

Feature engineering

Model training

Prediction

## Notes

Modular imports make it easy to pick only the components you need.

All models and transformers are pure NumPy, so you can inspect and extend them easily.

Designed for learning, experimentation, and building pipelines from scratch.