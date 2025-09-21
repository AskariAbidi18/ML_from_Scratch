import numpy as np 
from baremetalml.base.transformer import BaseTransformer
class OneHotEncoder(BaseTransformer):
    def fit(self, X):
        X = self.check_x(X)
        self.categories = [np.unique(X[:, i]) for i in range(X.shape[1])]
        return self 

    def transform(self, y):
        y = self.check_x(y)
        encoded_columns = list()
        for i in range(y.shape[1]):
            column = y[:, i]
            categories = self.categories[i]
            one_hot = np.zeros((y.shape[0], len(categories)), dtype=int)
            for j, category in enumerate(categories):
                one_hot[:, j] = (column == category).astype(int)
            encoded_columns.append(one_hot)
        return np.column_stack(encoded_columns)
    