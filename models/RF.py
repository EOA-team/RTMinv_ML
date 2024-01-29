'''
Random Forest Regressor model to perform a RTM inversion

@author Selene Ledain
'''

from sklearn.model_selection import cross_val_score, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import pandas as pd
from typing import Any, Dict, List, Optional
import pickle
import numpy as np
from tqdm import tqdm

class RF(RandomForestRegressor):
    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2, min_samples_leaf=1, random_state=None, k_folds: Optional[int] = 5):
        '''
        Random Forest regressor model

        :param n_estimators: number of trees in the forest
        :param max_depth: maximum depth of the tree
        :param min_samples_split: minimum number of samples required to split an internal node
        :param min_samples_leaf: minimum number of samples required to be at a leaf node
        :param random_state: 
        :param k_folds: number of folds for cross validation
        '''
        super().__init__(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state
        )
        self.k_folds = k_folds

    def fit(self, 
        X: pd.DataFrame,
        y: pd.Series) -> None:
        '''
        Fit the model on the training set, using k-fold cros validation if possible

        :param X: training data
        :param y: training labels
        '''
        if self.k_folds == 0:
            # If k_folds is 0, perform a standard training without k-fold cross-validation
            super().fit(X, y)
        else:
            # Perform k-fold cross-validation on the training data
            kf = KFold(n_splits=self.k_folds, shuffle=True, random_state=self.random_state)

            # List to store cross-validation RMSE scores
            cv_rmse_scores = []

            # Loop through each fold
            for train_idx, val_idx in tqdm(kf.split(X), desc ="Training CV fold"):
                X_train_fold, X_val_fold = X[train_idx], X[val_idx]
                y_train_fold, y_val_fold = y[train_idx], y[val_idx]

                # Fit the model on the training fold
                super().fit(X_train_fold, y_train_fold)

                # Make predictions on the validation fold
                y_val_pred = super().predict(X_val_fold)

                # Calculate and store RMSE for the current fold
                fold_rmse = mean_squared_error(y_val_fold, y_val_pred)**0.5
                cv_rmse_scores.append(fold_rmse)

        # Print cross-validation results
        print(f'Cross-Validation RMSE for each fold: {cv_rmse_scores}')
        print(f'Mean CV RMSE: {sum(cv_rmse_scores)/len(cv_rmse_scores)}')


    def predict(self, X_test: np.array) -> np.array:
        '''
        Make predictions on test set

        :param X_test: training data
        '''
        # Make predictions on the testing set
        return super().predict(X_test)
        

    def test_scores(self, y_test: pd.Series, y_pred: np.array) -> None:
        '''
        Compute scores on test set

        :param y_test: test labels
        :param y_pred: test predictions
        '''
        # Compute RMSE on the test set
        test_rmse = mean_squared_error(y_test, y_pred, squared=False)
        print(f'Test RMSE: {test_rmse}')

    def save(self, model, model_filename: str) -> None:
        ''' 
        Save trained model

        :param model: trained model
        :param model_filename: path to save model as .pkl file
        '''
        # Save the trained model to a file using pickle
        with open(model_filename, 'wb') as file:
            pickle.dump(model, file)
        print(f'Trained model saved to {model_filename}')
