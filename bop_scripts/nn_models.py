from random import sample
from sklearn.base import BaseEstimator
from sklearn.metrics import SCORERS
from sklearn.model_selection import train_test_split
from sklearn.utils import check_X_y, check_array
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from scipy.sparse import issparse
import numpy as np

class torchMLP (nn.Module):
    """
        Neural network model for 
    """

    def __init__(self, n_features, n_labels):
        super().__init__()

        self.network = nn.Sequential(*[
            nn.Linear(n_features, 200),
            nn.ReLU(),
            nn.Linear(200, 50),
            nn.ReLU(),
            nn.Linear(50, n_labels),
            nn.Sigmoid()
        ])

    def forward(self, x):
        
        y_hat = self.network(x)

        return y_hat

class torchMLPClassifier_sklearn (BaseEstimator):

    """
        Pytorch neural network with a sklearn-like API
    """

    def __init__ (self, model, n_epochs=50, early_stop=True, early_stop_metric="accuracy", early_stop_validations_size=0.1, batch_size=1024, learning_rate=1e-3, class_weight=None, device_train="cpu", device_predict="cpu"):
        """
            Parameters:
            -----------
            model: non instanciated pytorch neural network model with a n_features and n_labels parameter
            n_epochs: int, number of epochs
            early_stop: boolean, if true an evaluation dataset is created and used to stop the training
            early_stop_metric: str, metric score to evaluate the model, according to sklearn.metrics.SCORERS.keys()
            early_stop_validations_size: int or float, if float percentage of the train dataset used for validation, otherwise number of sample to use 
            batch_size: int, size of the training batch
            learning_rate: float, Adam optimizer learning rate
            class_weight: dict or str, same as the sklearn API
            device_train: str, device on which to train
            device_predict: str, device on which to predict
        """

        self.model = model

        self.n_epochs = n_epochs
        if early_stop and (early_stop_metric is not None) and (early_stop_metric in SCORERS.keys()) and (isinstance(early_stop_validations_size, int) or isinstance(early_stop_validations_size, float)):
            self.early_stop = early_stop
            self.early_stop_metric = SCORERS[early_stop_metric]
            self.early_stop_validations_size = early_stop_validations_size
        else:
            self.early_stop = False

        self.class_weight = class_weight
        self.learning_rate = learning_rate
        self.device_train = device_train
        self.device_predict = device_predict
        self.batch_size = batch_size

    def fit(self, X, y):
        """
            Training the model

            Parameters:
            -----------
            X_test: pandas dataframe of the features
            y_test: pandas dataframe of the labels
        """

        X, y = check_X_y(X, y, accept_sparse=True, multi_output=True)
        if y.ndim == 1:
            y = np.expand_dims(y, 1)

        # Validation split if early stopping
        if self.early_stop:
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=self.early_stop_validations_size)
            if issparse(X_val): # To deal with the sparse matrix situations
                X_val = X_val.toarray()
        else:
            X_train, y_train = X, y

        n_samples = y_train.shape[0]
        n_labels_values = len(np.unique(y_train))
        n_labels = y_train.shape[1]
        n_features = X.shape[1]

        # Raising the model
        self.network = self.model(n_features=n_features, n_labels=n_labels)
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.learning_rate)


        # Creating dataloader for X_train, y_train
        data_loader = DataLoader(range(X_train.shape[0]), shuffle=True, batch_size=self.batch_size)

        # Initializing loss function
        ## Getting weights
        if self.class_weight is not None:
            if self.class_weight == "balanced":
                weights = n_samples/(n_labels_values*np.bincount(y_train[:,0]))
                weights_dict = dict(zip(range(len(weights)), weights))
            else:
                weights_dict = self.class_weight
        else:
            weights_dict = None

        criterion = nn.BCELoss()

        # Running train
        last_score = 0
        for i in range(self.n_epochs):

            # Starting an epoch
            for indices in data_loader:
                self.optimizer.zero_grad()

                X_train_sample, y_train_sample = X_train[indices, :], y_train[indices, :]
                if issparse(X_train_sample): # To deal with the sparse matrix situations
                    X_train_sample = X_train_sample.toarray()
                X_train_sample_tensor, y_train_sample_tensor = [torch.tensor(x, dtype=torch.float32).to(self.device_train) for x in [X_train_sample, y_train_sample]]

                # Weighting the loss
                if self.class_weight is not None:
                    sample_weights = y_train_sample.copy()
                    for x, y in weights_dict.items():
                        sample_weights[sample_weights == x] = y
                    criterion.weigths = sample_weights

                # Get prediction
                y_train_sample_hat = self.network(X_train_sample_tensor)

                loss = criterion(y_train_sample_hat, y_train_sample_tensor)
                loss.backward()

                self.optimizer.step()

            # End of the Epoch : evaluating the score
            if self.early_stop:
                score = self.early_stop_metric(self, X_val, y_val)

                if score < last_score:
                    return self
                else:
                    last_score = score

        return self

    def predict(self, X):
        """
            Getting the prediction

            Parameters:
            -----------
            X_test: pandas dataframe of the features
        
        """

        y_hat_proba = self.predict_raw_proba(X)
        y_hat = (y_hat_proba >= 0.5)*1

        return y_hat

    def predict_raw_proba(self, X):
        """
            Getting the prediction score in tensor format

            Parameters:
            -----------
            X_test: pandas dataframe of the features
        
        """

        X = check_array(X, accept_sparse=True)
        if issparse(X): # To deal with the sparse matrix situations
            X = X.toarray()

        with torch.no_grad():
            model_predict = self.network.to(self.device_predict)
            model_predict.eval()

            # Create a tensor from X
            X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device_predict)
            
            y_hat_proba_torch = model_predict(X_tensor)
            y_hat_proba_torch = y_hat_proba_torch.detach().cpu().numpy()

        return y_hat_proba_torch

    def predict_proba(self, X):
        """
            Getting the prediction score in sklearn format

            Parameters:
            -----------
            X_test: pandas dataframe of the features
        
        """
        
        y_hat_proba_torch = self.predict_raw_proba(X)
        y_hat_proba_torch = np.concatenate([
            1-y_hat_proba_torch,
            y_hat_proba_torch
        ], axis=0)
        y_hat_proba = y_hat_proba_torch

        return y_hat_proba
