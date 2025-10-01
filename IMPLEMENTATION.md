# BareMetalML Implementation Guide

This document provides a **deep dive** into the **mathematical foundations** and **practical implementation** of BareMetalML components.  
It is intended for **learning, experimentation, and educational purposes**.

All classes are **modular**, so you can import them directly:

```python
from baremetalml import LinearRegression, StandardScaler, KNNClassifier
```

## 1. Base Classes

### 1.1 BaseModel

**Purpose**: Abstract class for all models with common interfaces and input validation.

**Responsibilities**:

*fit(X, y)* – Train the model

*predict(X)* – Make predictions

**Input validation**: *check_x_y* and *check_x*

**Code Snippet**:

```python
class BaseModel:
    def fit(self, X, y):
        raise NotImplementedError
    def predict(self, X):
        raise NotImplementedError

```

**Why it matters**: Ensures consistency and reduces repetitive code across models.

### 1.2 BaseTransformer

**Purpose**: Abstract class for all data transformers.

**Methods**:

*fit(X)* – Learn parameters from data

*transform(X)* – Apply transformation

*fit_transform(X)* – Combines fit + transform

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

## 2. Linear Regression

### 2.1 Mathematical Formulation

**Linear regression predicts**:

𝑦
^
=
𝑋
𝛽
+  
𝜖
y
^
​
 =Xβ+ϵ

*Where*:

𝑋
∈
𝑅
𝑛
×
𝑑
X∈R 
n×d
  = input matrix

𝛽
∈
𝑅
𝑑
β∈R 
d
  = weights

𝜖
ϵ = error

**Mean Squared Error (MSE)**:

*MSE*
=
1
𝑛
∑
𝑖
=
1
𝑛
(
𝑦
𝑖
−
𝑦
^
𝑖
)
2
MSE= 
n
1
​
  
i=1
∑
n
​
 (y 
i
​
 − 
y
^
​
  
i
​
 ) 
2
 
**Normal Equation (Analytical solution):**

𝛽
^
=
(
𝑋
𝑇
𝑋
)
−
1
𝑋
𝑇
𝑦
β
^
​
 =(X 
T
 X) 
−1
 X 
T
 y
Gradient Descent (Iterative solution):

𝛽
:
=
𝛽
−
𝛼
1
𝑛
𝑋
𝑇
(
𝑋
𝛽
−
𝑦
)
β:=β−α 
n
1
​
 X 
T
 (Xβ−y)
Where 
𝛼
α = learning rate.

2.2 Implementation in BareMetalML
python
Copy code
lr = LinearRegression(method="gradient_descent", learning_rate=0.01, n_iterations=1000)
lr.fit(X, y)
y_pred = lr.predict(X)
Highlights:

Supports Normal Equation & Gradient Descent

Automatically handles bias/intercept

Computes predictions as:

𝑦
^
=
𝑋
⋅
weights
+
bias
y
^
​
 =X⋅weights+bias
3. Logistic Regression
3.1 Mathematical Formulation
Sigmoid function:

𝜎
(
𝑧
)
=
1
1
+
𝑒
−
𝑧
σ(z)= 
1+e 
−z
 
1
​
 
Prediction:

𝑦
^
=
𝜎
(
𝑋
𝛽
)
y
^
​
 =σ(Xβ)
Binary Cross-Entropy Loss:

𝐿
(
𝛽
)
=
−
1
𝑛
∑
𝑖
=
1
𝑛
[
𝑦
𝑖
log
⁡
(
𝑦
^
𝑖
)
+
(
1
−
𝑦
𝑖
)
log
⁡
(
1
−
𝑦
^
𝑖
)
]
L(β)=− 
n
1
​
  
i=1
∑
n
​
 [y 
i
​
 log( 
y
^
​
  
i
​
 )+(1−y 
i
​
 )log(1− 
y
^
​
  
i
​
 )]
Gradient Descent Updates:

𝛽
:
=
𝛽
−
𝛼
1
𝑛
𝑋
𝑇
(
𝑦
^
−
𝑦
)
β:=β−α 
n
1
​
 X 
T
 ( 
y
^
​
 −y)
3.2 Implementation
python
Copy code
logr = LogisticRegression(n_iterations=1000, learning_rate=0.01)
logr.fit(X, y)
y_pred = logr.predict(X)
Computes probabilities using sigmoid

Updates weights via gradient of cross-entropy loss

Predicts 0/1 based on 0.5 threshold

4. K-Nearest Neighbors (KNN)
4.1 Mathematical Formulation
Distance Metrics:

Euclidean: 
𝑑
=
∑
(
𝑥
𝑖
−
𝑥
𝑗
)
2
d= 
∑(x 
i
​
 −x 
j
​
 ) 
2
 
​
 

Manhattan: 
𝑑
=
∑
∣
𝑥
𝑖
−
𝑥
𝑗
∣
d=∑∣x 
i
​
 −x 
j
​
 ∣

Minkowski: 
𝑑
=
(
∑
∣
𝑥
𝑖
−
𝑥
𝑗
∣
𝑝
)
1
/
𝑝
d=(∑∣x 
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

Classification: majority vote of k nearest neighbors

Regression: mean of k nearest neighbors

4.2 Implementation
python
Copy code
knn = KNNClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
Pipeline Illustration:

rust
Copy code
X_test -> compute distances -> select k nearest neighbors -> predict majority class
5. Transformers
5.1 StandardScaler
Equation:

𝑋
𝑠
𝑐
𝑎
𝑙
𝑒
𝑑
=
𝑋
−
𝜇
𝜎
X 
scaled
​
 = 
σ
X−μ
​
 
5.2 NormalScaler
Equation:

𝑋
𝑛
𝑜
𝑟
𝑚
=
𝑋
−
𝑋
𝑚
𝑖
𝑛
𝑋
𝑚
𝑎
𝑥
−
𝑋
𝑚
𝑖
𝑛
X 
norm
​
 = 
X 
max
​
 −X 
min
​
 
X−X 
min
​
 
​
 
5.3 LabelEncoder
Maps categorical labels to integers.
Example: {'cat':0, 'dog':1, 'bird':2}

5.4 OneHotEncoder
Converts categories to one-hot vectors.
Example:

[
′
𝑟
𝑒
𝑑
′
,
′
𝑏
𝑙
𝑢
𝑒
′
]
→
[
1
0
0
1
]
[ 
′
 red 
′
 , 
′
 blue 
′
 ]→[ 
1
0
​
  
0
1
​
 ]
5.5 PolynomialFeatures
Generates all polynomial combinations up to degree 
𝑑
d:

(
𝑥
1
,
𝑥
2
)
→
(
1
,
𝑥
1
,
𝑥
2
,
𝑥
1
2
,
𝑥
1
𝑥
2
,
𝑥
2
2
)
(x 
1
​
 ,x 
2
​
 )→(1,x 
1
​
 ,x 
2
​
 ,x 
1
2
​
 ,x 
1
​
 x 
2
​
 ,x 
2
2
​
 )
python
Copy code
poly = PolynomialFeatures(degree=2, include_bias=True)
X_poly = poly.fit_transform(X)
6. Example Pipeline
python
Copy code
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
Pipeline Overview:

rust
Copy code
Raw Data -> Scaling -> Polynomial Feature Expansion -> Linear Regression -> Predictions
Notes
All models and transformers are pure NumPy, easy to inspect and extend

Designed for learning, experimentation, and building pipelines from scratch

Modular imports make it simple to use any component:

python
Copy code
from baremetalml import LinearRegression, StandardScaler, KNNClassifier