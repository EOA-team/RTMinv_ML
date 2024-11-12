from models.RF import RF
from models.NN import NeuralNetworkRegressor
from models.NN_copy import NeuralNetworkRegressor2
from models.RidgeReg import RidgeReg
from models.LassoReg import LassoReg
from models.OLS import LinearReg
from models.SVR import SVRReg
#from models.GPR import GaussianProcessActiveLearner


MODELS = {
    "RF": RF,
    "NN": NeuralNetworkRegressor,
    "NN2": NeuralNetworkRegressor2,
    "RidgeReg": RidgeReg,
    "LassoReg": LassoReg,
    "OLS": LinearReg,
    "SVR": SVRReg,
    #"GPR": GaussianProcessActiveLearner
}