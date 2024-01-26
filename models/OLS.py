'''
Linear Regression model to perform a RTM inversion

@author Selene Ledain
'''

from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pandas as pd
from typing import Optional
import pickle
import numpy as np

class LinearReg(LinearRegression):
    def __init__(self, fit_intercept=True, copy_X=True, n_jobs=None, k_folds: Optional[int] = 5):
        '''
        Linear Regression model

        :param fit_intercept: Whether to calculate the intercept for this model.
        :param copy_X: Whether to copy X in fit method.
        :param n_jobs: Number of CPU cores to use for computation.
        :param k_folds: Number of folds for cross-validation
        '''
        super().__init__(
            fit_intercept=fit_intercept,
            copy_X=copy_X,
            n_jobs=n_jobs
        )
        self.k_folds = k_folds

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        '''
        Fit the model on the training set, using k-fold cross-validation if possible

        :param X: Training data
        :param y: Training labels
        '''
        if self.k_folds == 0:
            # If k_folds is 0, perform standard training without k-fold cross-validation
            super().fit(X, y)
        else:
            # Perform k-fold cross-validation on the training data
            kf = KFold(n_splits=self.k_folds, shuffle=True, random_state=None)  # Set random_state as needed

            # List to store cross-validation RMSE scores
            cv_rmse_scores = []

            # Loop through each fold
            for train_idx, val_idx in kf.split(X):
                X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
                y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]

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

    def predict(self, X_test: pd.DataFrame) -> np.array:
        '''
        Make predictions on the test set

        :param X_test: Test data
        '''
        # Make predictions on the testing set
        return super().predict(X_test)


    def test_scores(self, y_test: pd.Series, y_pred: np.array) -> None:
        '''
        Compute scores on the test set

        :param y_test: Test labels
        :param y_pred: Test predictions
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