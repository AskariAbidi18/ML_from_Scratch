import numpy as np

def accuracy(y_true, y_pred):
    TP = TN = FP = FN = 0
    for true, pred in zip(y_true, y_pred):
        if true and pred:
            TP += 1
        elif not true and not pred:
            TN += 1
        elif not true and pred:
            FP += 1
        elif true and not pred:
            FN += 1
    return (TP + TN) / len(y_true)

def precision(y_true, y_pred):
    TP = FP = 0
    for true, pred in zip(y_true, y_pred):
        if true and pred:
            TP += 1
        elif not true and pred:
            FP += 1
    return TP / (TP + FP) if (TP + FP) > 0 else 0

def recall(y_true, y_pred):
    TP = FN = 0
    for true, pred in zip(y_true, y_pred):
        if true and pred:
            TP += 1
        elif true and not pred:
            FN += 1
    return TP / (TP + FN) if (TP + FN) > 0 else 0

def f1_score(y_true, y_pred):
    TP = FP = FN = 0
    for true, pred in zip(y_true, y_pred):
        if true and pred:
            TP += 1
        elif not true and pred:
            FP += 1
        elif true and not pred:
            FN += 1
    precision_val = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall_val = TP / (TP + FN) if (TP + FN) > 0 else 0
    return 2 * (precision_val * recall_val) / (precision_val + recall_val) if (precision_val + recall_val) > 0 else 0

def confusion_matrix(y_true, y_pred):
    TP = TN = FP = FN = 0
    for true, pred in zip(y_true, y_pred):
        if true and pred:
            TP += 1
        elif not true and not pred:
            TN += 1
        elif not true and pred:
            FP += 1
        elif true and not pred:
            FN += 1
    return np.array([[TP, FP],
                     [FN, TN]])
