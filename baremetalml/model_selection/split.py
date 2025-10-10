import numpy as np
def train_test_split(X, y, test_size=0.2, random_state=None, shuffle=True, stratify=None):
    X = np.array(X)
    y = np.array(y)

    n = len(X)
    if random_state is not None:
        np.random.seed(random_state)

    if test_size < 1:
        n_test = int(n * test_size)
    else:
        n_test = int(test_size)
    n_train = n - n_test
    if stratify is None:
        indices = np.arange(n)
        if shuffle:
            np.random.shuffle(indices)

        test_idx = indices[:n_test]
        train_idx = indices[n_test:]

    else:
        train_idx = []
        test_idx = []

        classes, y_indices = np.unique(y, return_inverse=True)

        for cls in classes:
            cls_indices = np.where(y_indices == cls)[0]
            if shuffle:
                np.random.shuffle(cls_indices)
            n_cls_test = int(len(cls_indices) * test_size) if test_size < 1 else int(test_size * len(cls_indices) / n)
        
            test_idx.extend(cls_indices[:n_cls_test])
            train_idx.extend(cls_indices[n_cls_test:])

        train_idx = np.array(train_idx)
        test_idx = np.array(test_idx)

        if shuffle:
            np.random.shuffle(train_idx)
            np.random.shuffle(test_idx)

    X_train = X[train_idx]
    X_test = X[test_idx]

    y_train = y[train_idx]
    y_test = y[test_idx]    
    
    return X_train, X_test, y_train, y_test
