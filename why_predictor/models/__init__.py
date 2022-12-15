"""Models"""
from enum import Enum

from .abstract_model import BasicModel
from .knn_regression import (
    ChainedKNNRegressor,
    MultioutputKNNRegressor,
    ShiftedKNNRegressor,
)
from .linear_regression import (
    ChainedLinearRegressor,
    MultioutputLinearRegressor,
    ShiftedLinearRegressor,
)
from .mlp_regressor import (
    MultiLayerPerceptronRegressor,
    MultioutputMLPRegressor,
)
from .random_forest_regression import MultioutputRFRegressor, RFRegressor
from .sgd_regressor import (
    MultioutputSGDRegressor,
    StochasticGradientDescentRegressor,
)
from .svm_regression import MultioutputSVMRegressor, SupportVectorRegressor
from .tree_regression import (
    DecissionTreeRegressor,
    MultioutputDecissionTreeRegressor,
)


class Models(Enum):
    """Enum with Models"""

    SHIFT_LR = ShiftedLinearRegressor
    RF = RFRegressor
    SHIFT_KNN = ShiftedKNNRegressor
    DT = DecissionTreeRegressor
    SVR = SupportVectorRegressor
    SGD = StochasticGradientDescentRegressor
    MLP = MultiLayerPerceptronRegressor
    CHAIN_LR = ChainedLinearRegressor
    CHAIN_KNN = ChainedKNNRegressor
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
