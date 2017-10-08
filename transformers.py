import numpy as np

from sklearn.base import TransformerMixin
from sklearn.base import BaseEstimator
from sklearn.preprocessing import Normalizer

class FilterSimu(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.9):
        self.threshold = threshold
        return

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        for i in range(X.shape[0]):
            e = np.mean(X[i, :])
            d = np.std(X[i, :])
            thresh = min(e + self.threshold*d, np.max(X[i, :]))
            for j in range(X.shape[1]):
                if X[i, j] < thresh:
                    X[i, j] = 0
                else:
                    X[i, j] = 1
        #X = Normalizer(norm='l1').fit_transform(X)
        return X


class LabelConverter(object):
    def __init__(self, threshold=1.0):
        self.threshold = threshold
        return

    def fit(self, y):
        self.mean = np.mean(y)
        self.std = np.std(y)
        return

    def transform(self, y):
        label = np.zeros_like(y, dtype=int)
        #thresholds = [self.mean - self.threshold*self.std, self.mean + self.threshold*self.std]
        thresholds = [self.mean]

        for i in range(y.shape[0]):
            label[i] = len(thresholds)
            for j, threshold in enumerate(thresholds):
                if y[i] < threshold:
                    label[i] = j
                    break
        return label

    def fit_transform(self, y):
        self.fit(y)
        return self.transform(y)