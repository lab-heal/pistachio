import numpy as np
from sklearn.base import BaseEstimator


class HrModel(BaseEstimator):
    def __init__(self, low=105, high=129):
        self.low = low
        self.high = high

    def fit(self, X, y):
        return self

    def predict(self, X):
        mean_hr = X["hr_moving_avg_10m"]
        return (mean_hr > self.low) & (mean_hr < self.high)

    def predict_proba(self, X):
        preds = self.predict(X)
        proba = np.zeros((len(X), 2))
        proba[:, 1] = preds.astype(float)
        proba[:, 0] = 1 - proba[:, 1]
        return proba
