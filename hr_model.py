import numpy as np
from sklearn.base import BaseEstimator


class HrModel(BaseEstimator):
    def fit(self, X, y):
        return self

    def predict(self, X):
        mean_hr = X["hr_moving_avg_10m"]
        return (mean_hr > 105) & (mean_hr < 129)

    def predict_proba(self, X):
        preds = self.predict(X)
        proba = np.zeros((len(X), 2))
        proba[:, 1] = preds.astype(float)
        proba[:, 0] = 1 - proba[:, 1]
        return proba
