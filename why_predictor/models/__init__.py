"""Models"""
from enum import Enum

from .abstract_model import BasicModel
from .knn_regression import KNNRegressor, MultioutputKNNRegressor
from .linear_regression import LinearRegressor, MultioutputLinearRegressor
from .mlp_regressor import (
    MultiLayerPerceptronRegressor,
    MultioutputMLPRegressor,
)
from .random_forest_regression import MultioutputRFRegressor, RFRegressor
from .sgd_regressor import (
    StochasticGradientDescentRegressor,
    MultioutputSGDRegressor,
)
from .svm_regression import SupportVectorRegressor, MultioutputSVMRegressor
from .tree_regression import (
    DecissionTreeRegressor,
    MultioutputDecissionTreeRegressor,
)


class Models(Enum):
    """Enum with Models"""

    LR = LinearRegressor
    RF = RFRegressor
    KNN = KNNRegressor
    DT = DecissionTreeRegressor
    SVR = SupportVectorRegressor
    SGD = StochasticGradientDescentRegressor
    MLP = MultiLayerPerceptronRegressor
    MULTI_LR = MultioutputLinearRegressor
    MULTI_RF = MultioutputRFRegressor
    MULTI_KNN = MultioutputKNNRegressor
    MULTI_DT = MultioutputDecissionTreeRegressor
    MULTI_SVR = MultioutputSVMRegressor
    MULTI_SGD = MultioutputSGDRegressor
    MULTI_MLP = MultioutputMLPRegressor


__all__ = [
    "BasicModel",
    "Models",
]
