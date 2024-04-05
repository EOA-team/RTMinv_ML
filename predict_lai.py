"""
Apr 5th 2024

@author: Sélène Ledain 
"""

import numpy as np
import pickle
import matplotlib.pyplot as plt

def predict(arr: np.array, model_path: str) -> np.array:
  """
  Load data and model, make predictions

  :param: arr: 4D numpy array (x, y, bands, date)
  :param: model_path: path to pickled model
  :returns: LAI predictions in shape (x, y)
  """
  # Load model
  with open(model_path, 'rb') as f:
    model = pickle.load(f)
  # Open corresponding scaler to normalise data
  with open(model_path.split('.')[0] + '_scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
  
  # Normalise the data
  arr_reshaped = arr.reshape(-1, arr.shape[2]) # Reshape the image to  (width * height * date, bands)
  arr_reshaped_norm = scaler.transform(arr_reshaped)
  arr_norm = arr_reshaped_norm.reshape(arr.shape) # Reshape back to original
  arr_norm[arr==0] = 0
  
  # Make predictions 
  preds = np.zeros((arr.shape[0], arr.shape[1], arr.shape[-1])) # Initialize array to store predictions

  for n in range(arr_norm.shape[-1]):
    for i in range(arr_norm.shape[0]): 
        for j in range(arr_norm.shape[1]): 

            pixel_features = arr_norm[i, j, :, n]
            if pixel_features.sum() == 0:
                # Outside of fields
                preds[i, j, n] = 0
                continue
            else:
                pixel_features = pixel_features.reshape(1, -1)
                lai = model.predict(pixel_features)
                preds[i, j, n] = lai
  
  return preds

if __name__ == '__main__':

  data_path = 's2_data_3.npz'
  model_path = 'NN_soilscaled2_inverse10.pkl'
  output_path = 'lai_preds.npz'

  #############
  # LOAD DATA
  variables = np.load(data_path)
  variables.allow_pickle=True
  locals().update(variables)
  del variables
  # the images are in im which is 4-D [y,x,bands,image num]
  # bands are ['B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B11', 'B12', 'SCL'] 

  #############
  # PREDICT
  preds = predict(im[:,:,:-1,:], model_path)

  #############
  # SAVE
  np.savez(output_path, preds=preds) # predictions saved in preds

  #############
  # PLOT AN EXAMPLE
  plt.figure()
  plt.imshow(preds[:,:,0], cmap='Greens')
  cbar = plt.colorbar()
  cbar.set_label('LAI')
  plt.title('Predicted LAI')
  plt.savefig(f'predstest.png')

