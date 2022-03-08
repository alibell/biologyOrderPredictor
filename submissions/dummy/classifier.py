import numpy as np
from sklearn.base import BaseEstimator


class Classifier(BaseEstimator):

    def fit(self, X, y):
        self.pred = y.mean(axis=0)

        return self

    def predict_proba(self, X):
        y_pred = np.repeat(self.pred.reshape(1, -1), X.shape[0], axis=0)

        return y_pred

    def predict(self, X):
        y_pred = self.predict_proba(X)

        return (y_pred >= 0.5)*1