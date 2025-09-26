# -----------------------------
# Base classes
# -----------------------------
from .base.model import BaseModel
from .base.transformer import BaseTransformer

# -----------------------------
# Linear models
# -----------------------------
from .linear import LinearRegression, LogisticRegression

# -----------------------------
# Neighbours (KNN)
# -----------------------------
from .neighbours import KNNClassifier, KNNRegressor

# -----------------------------
# Preprocessing
# -----------------------------
from .preprocessing import (
    LabelEncoder,
    OneHotEncoder,
    NormalScaler,
    StandardScaler,
    PolynomialFeatures
)

# -----------------------------
# Expose all in package level
# -----------------------------
__all__ = [
    # Base
    "BaseModel",
    "BaseTransformer",
    
    # Linear models
    "LinearRegression",
    "LogisticRegression",
    
    # KNN
    "KNNClassifier",
    "KNNRegressor",
    
    # Preprocessing
    "LabelEncoder",
    "OneHotEncoder",
    "NormalScaler",
    "StandardScaler",
    "PolynomialFeatures"
]
