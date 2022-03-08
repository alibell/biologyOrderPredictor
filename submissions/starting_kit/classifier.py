import sys
import os
from pathlib import Path
sys.path.append(str(Path(os.path.dirname(__file__)).parent)) # Dirty but it works

from bop_scripts.preprocessing import remove_outliers
from bop_scripts.nn_models import torchMLPClassifier_sklearn, torchMLP
from bop_scripts.models import generate_model, fit_all_classifiers
import torch
import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator

qualitatives_variables = ["gender", "last_7", "last_30"]
quantitatives_variables = ['age', 'temperature', 'heartrate', 'resprate', 'o2sat', 'sbp', 'dbp', 'pain']
text_variables = ["chiefcomplaint"]
labels = ['Cardiaque', 'Coagulation', 'Gazometrie', 'Glycemie_Sanguine', 'Hepato-Biliaire', 'IonoC', 'Lipase', 'NFS', 'Phospho-Calcique']
variables_ranges = {
    "temperature":[60,130],
    "heartrate":[20, 300],
    "resprate":[5, 50],
    "o2sat":[20, 100],
    "sbp":[40, 250],
    "dbp":[20, 200],
    "pain":[0,10]
}
device = "cuda:0" if torch.cuda.is_available() else "cpu"

def torch_classifier_fn ():

    torch_classifier = torchMLPClassifier_sklearn(
        torchMLP,
        early_stop_validations_size=10000,
        early_stop=True,
        early_stop_metric="f1",
        early_stop_tol=1,
        n_epochs=50,
        device_train= device,
        device_predict="cpu",
        class_weight="balanced",
        learning_rate=1e-4,
        verbose=False
    )

    torch_sklearn_classifier = generate_model(
            torch_classifier,
            qualitatives_variables,
            quantitatives_variables,
            text_variables[0],
            remove_outliers=True,
            outliers_variables_ranges=variables_ranges,
            CountVectorizer_kwargs={"ngram_range":(1,1), "max_features":600}
    )

    return torch_sklearn_classifier

class Classifier(BaseEstimator):

    def preprocess (self, X, y=None):
        X_clean, outliers = remove_outliers(X, variables_ranges)
        if y is not None:
            y = pd.DataFrame(y, columns=labels)

        return X_clean, y

    def fit(self, X, y):
        X, y = self.preprocess(X, y)
        self.classifiers = fit_all_classifiers(
            torch_classifier_fn,
            X,
            y,
            verbose=False
        )
        return self

    def predict_proba(self, X):
        X, y = self.preprocess(X)
        predictions = []
        y_columns = labels
        for y_column in y_columns:
            predictions.append(self.classifiers[y_column].predict_proba(X)[:,1].reshape(-1, 1))
        y_pred = np.concatenate(predictions, axis=1)

        return y_pred

    def predict(self, X):
        y_pred = self.predict_proba(X)

        return (y_pred >= 0.5)*1