import numpy as np
from baremetalml import BaseModel
class GaussianNB(BaseModel):
    def  __init__(self):
        self.var_smoothing = 1e-9

    def fit(self, X, y):
        classes = np.unique(y)
        

