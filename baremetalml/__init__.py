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
# Model selection
# -----------------------------
from .model_selection import train_test_split

# -----------------------------
# Metrics
# -----------------------------
from .metrics import MAE, MSE, RMSE, R2, accuracy, precision, recall, f1_score, confusion_matrix

# -----------------------------
# Naive Bayes
# -----------------------------
from .naive_bayes import GaussianNB, MultinomialNB

# -----------------------------
# Support Vector Machines
# -----------------------------
from .SVM import SupportVectorClassifier, SupportVectorRegressor

# -----------------------------
# Decision Trees
# -----------------------------
from .tree import DecisionTreeClassifier, DecisionTreeRegressor

# -----------------------------
# Expose all in package level
# -----------------------------
__all__ = [
    # Base
    "BaseModel", "BaseTransformer",
    # Linear models
    "LinearRegression", "LogisticRegression",
    # KNN
    "KNNClassifier", "KNNRegressor",
    # Preprocessing
    "LabelEncoder", "OneHotEncoder", "NormalScaler", "StandardScaler", "PolynomialFeatures",
    # Model selection
    "train_test_split",
    # Metrics
    "MAE", "MSE", "RMSE", "R2", "accuracy", "precision", "recall", "f1_score", "confusion_matrix",
    # Naive Bayes
    "GaussianNB", "MultinomialNB",
    # SVM
    "SupportVectorClassifier", "SupportVectorRegressor",
    # Decision Trees
    "DecisionTreeClassifier", "DecisionTreeRegressor"
]
