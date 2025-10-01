# BareMetalML

**BareMetalML** is a lightweight, modular machine learning library built from scratch in Python by **Askari Abidi**.  
It is designed for **learning, experimentation, and educational purposes**, providing a clear and simple interface for core machine learning algorithms and preprocessing tools.  
All models and transformers are implemented using **pure NumPy**, making the library easy to understand and extend.

---

## Features

### Easy and Modular Imports
BareMetalML allows **modular imports**, so you don’t need to worry about the internal file structure.  
For example, you can import any model or transformer like this:

```python
from baremetalml import LinearRegression, StandardScaler

No need to know which file the class is defined in — everything is ready to use.

Core Models
Linear Regression – Ordinary Least Squares (Normal Equation) and Gradient Descent methods.

Logistic Regression – Binary classification with Gradient Descent optimization.

K-Nearest Neighbors (KNN) – Classification and Regression with multiple distance metrics.

Preprocessing and Transformers
StandardScaler – Standardizes features to mean=0, standard deviation=1.

NormalScaler – Min-max scaling to [0,1].

LabelEncoder – Converts categorical labels to integers.

OneHotEncoder – Converts categorical features into one-hot encoded vectors.

PolynomialFeatures – Generates polynomial and interaction features up to a given degree.

Base Classes
BaseModel – Abstract base class for all models with input validation.

BaseTransformer – Abstract base class for all transformers with fit, transform, and fit_transform interface.

What’s Inside
BareMetalML includes:

All basic regression and classification models you need to start learning ML from scratch.

Preprocessing tools for handling both numerical and categorical data.

Feature engineering utilities like polynomial feature generation.

Modular structure for easy imports and extensibility.

Installation
BareMetalML is available on PyPI. Install it with:

bash
Copy code
pip install baremetalml
Then import the modules you need:

python
Copy code
from baremetalml import LinearRegression, StandardScaler, KNNClassifier
Author
Askari Abidi

Contribute & Support
BareMetalML is an open-source educational project.
Feel free to contribute, suggest improvements, or report issues on the repository.

If you find it useful, you can support the project via donations or by sharing it with others.
Every contribution helps make learning ML from scratch easier and more accessible!