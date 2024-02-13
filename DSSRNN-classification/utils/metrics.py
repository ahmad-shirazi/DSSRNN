import numpy as np
from sklearn.metrics import confusion_matrix


def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    d += 1e-12
    return 0.01*(u / d).mean(-1)


def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))


def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))

def accuracy(true_positive, true_negative, false_positive, false_negative):
    total = true_positive + true_negative + false_positive + false_negative
    return (true_positive + true_negative) / total

def precision(true_positive, false_positive):
    return true_positive / (true_positive + false_positive)

def recall(true_positive, false_negative):
    return true_positive / (true_positive + false_negative)

def f1_score(precision, recall):
    return 2 * ((precision * recall) / (precision + recall))

def metric(pred, true):
    true_positive = np.sum(np.logical_and(pred == 1, true == 1))
    false_positive = np.sum(np.logical_and(pred == 1, true == 0))
    true_negative = np.sum(np.logical_and(pred == 0, true == 0))
    false_negative = np.sum(np.logical_and(pred == 0, true == 1))
    # CM = confusion_matrix(true, pred)

    # true_negative = CM[0][0]
    # false_negative = CM[1][0]
    # true_positive = CM[1][1]
    # false_positive = CM[0][1]

    acc = accuracy(true_positive, true_negative, false_positive, false_negative)
    prec = precision(true_positive, false_positive)
    rec = recall(true_positive, false_negative)
    f1 = f1_score(prec, rec)

    return acc, prec, rec, f1, true_positive, false_positive, true_negative, false_negative


# def metric(pred, true):
#     mae = MAE(pred, true)
#     mse = MSE(pred, true)
#     rmse = RMSE(pred, true)
#     mape = MAPE(pred, true)
#     mspe = MSPE(pred, true)
#     rse = RSE(pred, true)
#     corr = CORR(pred, true)

#     return mae, mse, rmse, mape, mspe, rse, corr
