#%%
from multiprocessing.sharedctypes import Value
import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__))) # Dirty but it works

import rampwf as rw
from rampwf.prediction_types.detection import Predictions as DetectionPredictions
from rampwf.utils.importing import import_module_from_source
import pandas as pd
import numpy as np
from sklearn.model_selection import ShuffleSplit, train_test_split
from sklearn.metrics import roc_auc_score
from bop_scripts import preprocessing
import itertools
# %%
# Parameters
problem_title = 'Biology Order Prescription'
data = "./data/mimic-iv.sqlite"
lab_dictionnary = pd.read_csv("./config/lab_items.csv").set_index("item_id")["3"].to_dict()
get_drugs, get_diseases = True, True
# %%
# Getting data
if os.path.exists("./data/X.csv"):
    X = pd.read_csv("./data/X.csv")
else:
    print("Creating X dataset (first run)")
    X = preprocessing.generate_features_dataset(
        database="./data/mimic-iv.sqlite",
        get_drugs=get_drugs,
        get_diseases=get_diseases
    )
    X["last_7"] = X["last_7"].fillna(0)
    X["last_30"] = X["last_30"].fillna(0)
    X.to_csv("./data/X.csv", header=True, index=False)

if os.path.exists("./data/y.csv"):
    y = pd.read_csv("./data/y.csv")
else:
    print("Creating y dataset (first run)")
    y = preprocessing.generate_labels_dataset(
        database="./data/mimic-iv.sqlite",
        lab_dictionnary=lab_dictionnary,
    )
    y.to_csv("./data/y.csv", header=True, index=False)

# Creating train and test
if (os.path.exists("./data/train.csv") == False) or (os.path.exists("./data/test.csv") == False):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=10000, random_state=42
    )

    train = pd.merge(
        X_train,
        y_train,
        left_on="stay_id",
        right_on="stay_id"
    ).reset_index(drop=True)
    train.to_csv("./data/train.csv", header=True, index=False)

    test = pd.merge(
        X_test,
        y_test,
        left_on="stay_id",
        right_on="stay_id"
    ).reset_index(drop=True)
    test.to_csv("./data/test.csv", header=True, index=False)

# %%
# Get rampwf evaluation 
class make_detection_fixed(DetectionPredictions):
    def __init__ (self, *args, **kwargs):
        super().__init__ (*args, **kwargs)

    def set_valid_in_train (self, predictions, test_is):
        self.y_pred = np.repeat(self.y_pred.reshape(-1, 1), predictions.y_pred.shape[1], axis=1)
        self.y_pred[test_is] = predictions.y_pred

    def set_slice(self, valid_indexes):
        if isinstance(valid_indexes, list):
            self.y_pred = self.y_pred[valid_indexes]

    @classmethod
    def combine(cls, predictions_list, index_list=None, greedy=False):
        if index_list is None:  # we combine the full list
            index_list = range(len(predictions_list))
        y_comb_list = [predictions_list[i].y_pred for i in index_list]

        n_preds = y_comb_list[0].shape[0]
        n_labels = y_comb_list[0].shape[1]

        y_preds_combined = np.empty((n_preds, n_labels), dtype=object)

        for i in range(n_preds):
            preds_list = [preds[i,:] for preds in y_comb_list
                          if preds[i, 0] is not None]

            if len(preds_list) == 1:
                # no overlap in the different prediction sets -> simply take
                # the single one that is not None
                preds_combined = preds_list[0]
            elif len(preds_list) > 1:
                preds_combined, _ = combine_predictions(
                    preds_list, cls.iou_threshold, greedy=greedy)

            if len(preds_list) > 0:
                y_preds_combined[i,:] = preds_combined

        combined_predictions = cls(y_pred=y_preds_combined)
        
        return combined_predictions

#%%
def combine_predictions(preds_list, iou_threshold, greedy=False):
    """
    Combine multiple sets of predictions (of different models)
    for a single patch.
    """

    combined_prediction = np.array(preds_list).mean(axis=0)
    return combined_prediction, None

#%%
_features_name = X.columns.tolist()[1:]
_prediction_label_names = y.columns.tolist()[1:]
prediction_type = make_detection_fixed
# %%
class ROCAUC_fixed(rw.score_types.base.BaseScoreType):
    is_lower_the_better = False
    minimum = 0.0
    maximum = 1.0

    def __init__(self, index, name='roc_auc', precision=2):
        self.name = name
        self.precision = precision
        self.index = index

    def score_function(self, ground_truths, predictions):
        """A hybrid score.
        It tests the predicted _probability_ of the second class
        against the true _label index_ (which is 0 if the first label is the
        ground truth, and 1 if it is not, in other words, it is the
        true probability of the second class). Thus we have to override the
        `Base` function here
        """

        y_proba = predictions.y_pred[:, self.index]
        y_true_proba = ground_truths.y_pred[:, self.index]
    
        mask = (y_proba != None)
        y_proba, y_true_proba = y_proba[mask], y_true_proba[mask]
        
        return self.__call__(y_true_proba, y_proba)

    def __call__(self, y_true_proba, y_proba):
        return roc_auc_score(y_true_proba, y_proba)

# %%
class customClassifier(rw.workflows.Classifier):
    def train_submission (self, module_path, X, y_array, train_is=None, prev_trained_model=None):
        if train_is is None:
            train_is = slice(None, None, None)
        
        classifier = import_module_from_source(
            os.path.join(module_path, self.element_names[0] + '.py'),
            self.element_names[0],
            sanitize=True
        )
        clf = classifier.Classifier()
        if prev_trained_model is None:
            clf.fit(X.iloc[train_is,:], y_array[train_is])
        else:
            clf.fit(
                X.iloc[train_is,:], y_array[train_is], prev_trained_model)

        return clf


workflow = customClassifier()
score_types = [ROCAUC_fixed(name=f"AUC {_prediction_label_names[i]}", index=i) for i in range(len(_prediction_label_names))]
Predictions = prediction_type
# %%
def get_cv(X, y):
    cv = ShuffleSplit(n_splits=3, test_size=0.2, random_state=42)
    return cv.split(X, y)


def _read_data(path, f_name):
    df = pd.read_csv(os.path.join(path, 'data', f_name))
    X = df[_features_name]
    y = df[_prediction_label_names].astype("int").values

    return X, y

def get_train_data(path='.'):
    f_name = 'train.csv'
    return _read_data(path, f_name)

def get_test_data(path='.'):
    f_name = 'test.csv'
    return _read_data(path, f_name)
