'''
Train a model to perform a RTM inversion

@author Selene Ledain
'''

import pandas as pd
from sklearn.model_selection import train_test_split
from datetime import datetime
from models import MODELS

#############################################
# DEFINE TRAINING PARAMS

data_path = '../results/lut_based_inversion/prosail_danner-etal_switzerland_lai-cab-ccc-car_lut_no-constraints.pkl'
train_cols = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08',
 'B8A', 'B09', 'B10', 'B11', 'B12']
target_col = 'lai'
test_size = 0.2
k_folds = 4
model_name = 'RF' # see possible models in models.__init__.py
model_filename = model_name +  datetime.now().strftime("%Y%m%d_%H%M%S") + '.pkl' # path to save trained model

# TO DO: model hyperparameters as input -> could be a grid of hyperparams 


#############################################
# SET UP DATA AND MODEL

df = pd.read_pickle(data_path)
X = df[train_cols]
y = df[target_col]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

# Initialize model based on the specified type
if model_name not in MODELS:
  raise ValueError(f"Invalid model type: {model_type}")
else:
  model = MODELS[model_name]()

  # Fit model, with k-fold cross validation if defined
  model.fit(X=X_train, y=y_train, k_folds=k_folds)

  # To do: hyperparam tuning -> or have a different script for tuning that calls the train , like earthnet minicuber

  # Test model
  y_pred = model.predict(X_test=X_test)

  # Compute scores and metrics
  model.test_scores(y_test=y_test, y_pred=y_pred)

  # Save model for future use
  model.save(model=model, model_filename=model_filename)



    

