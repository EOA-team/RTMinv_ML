# LAI retrieval

Retrieval of leaf area index (LAI) values by inverting PROSAIL radiative transfer model. Inversion is done by training a neural network on PROSAIL simulations.

## 1. Collect soil data across Switzerland

Using Google Earth Engine, produce yearly mean composites of bare soil pixels, for arable land. The Javascript code is provided in `S2_baresoil_GEE.txt`.\
Extract a dataset of bare soils usin K-means clustering with 
```
python bare_soil_GEE.py
```
