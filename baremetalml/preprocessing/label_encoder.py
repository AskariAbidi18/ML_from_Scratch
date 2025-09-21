import numpy as np 
from baremetalml.base.transformer import BaseTransformer
class LabelEncoder(BaseTransformer):
    def fit(self,y):
        y = self.check_x(y)
        self.classes_ = np.unique(y)
        self.class_to_index = {cls: idx for idx, cls in enumerate(self.classes_)}

    def transform(self, y, unknown_value=-1):
        y = self.check_x(y)
        return np.array([self.class_to_index.get(label, unknown_value) for label in y])
        
    def check_x(self, y):
        y = np.array(y)
        if y.ndim != 1:
            raise ValueError("y must be 1D (samples, )")
        return y
    
