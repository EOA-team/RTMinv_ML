from models.RF import RF
from models.NN import NeuralNetworkRegressor
from models.RidgeReg import RidgeReg
from models.LassoReg import LassoReg
from models.OLS import LinearReg
from models.SVR import SVRReg
from models.GPR import GaussianProcessActiveLearner


MODELS = {
    "RF": RF,
    "NN": NeuralNetworkRegressor,
    "RidgeReg": RidgeReg,
    "LassoReg": LassoReg,
    "OLS": LinearReg,
    "SVR": SVRReg,
    "GPR": GaussianProcessActiveLearner
}