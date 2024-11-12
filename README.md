# LAI retrieval

Retrieval of leaf area index (LAI) values by inverting PROSAIL radiative transfer model. Inversion is done by training a neural network on PROSAIL simulations.

## 1. Collect soil data across Switzerland

Using Google Earth Engine, produce yearly mean composites of bare soil pixels, for arable land. The Javascript code is provided in `S2_baresoil_GEE.txt`. The collected data is saved in `data/S2_baresoil_GEE.zip` (unzip folder before running code).\
Extract a dataset of bare soils using K-means clustering and save samples to .pkl file (`results/GEE_baresoil_v2/*.pkl`) with 
```
python bare_soil_GEE.py
```

## 2. PROSAIL forward runs

Launch PROSAIL forward runs with 
```
python simulate_S2_spectra_soil.py
```

The runs can be configured in the file directly. (We created 50k runs  with soil for Sentinel-2A and 2B each, and 50k each without soil. We then also created the test set of size 10k for Sentinel-2A and 2B.)\
The parametrisation of the RTM (with and without soil) is provided in `lut_params`.

Results of the forward runs are stored in `results/lut_based_inversion`.


## 3. Create validation set

Download S2 data corresponding to in-situ measurements with
```
python download_validation.py
```

## 4. Train model

To train the inversion model you need to configure the config files (`config/config_NN.yaml`). You specify the model and its parameters, the train, test and validation sets.
```
python train.py configs/config_NN.yaml
```

Then test on the validation data with 
```
python test.py configs/config_NN.yaml
```


## 5. Add noise

Add noise to the LUT training data using different models and at different amounts
```
python noise.py
```

## 6. Tune model
## 7. Results and predictions