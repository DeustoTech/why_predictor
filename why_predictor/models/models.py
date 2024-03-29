"""Models"""
from enum import Enum

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
    ChainedMultiLayerPerceptronRegressor,
    MultioutputMLPRegressor,
    ShiftedMultiLayerPerceptronRegressor,
)
from .random_forest_regression import (
    ChainedRFRegressor,
    MultioutputRFRegressor,
    ShiftedRFRegressor,
)
from .sgd_regressor import (
    ChainedStochasticGradientDescentRegressor,
    MultioutputSGDRegressor,
    ShiftedStochasticGradientDescentRegressor,
)
from .svm_regression import (
    ChainedSupportVectorRegressor,
    MultioutputSVMRegressor,
    ShiftedSupportVectorRegressor,
)
from .tree_regression import (
    ChainedDecissionTreeRegressor,
    MultioutputDecissionTreeRegressor,
    ShiftedDecissionTreeRegressor,
)


class Models(Enum):
    """Enum with Models"""

    SHIFT_LR = ShiftedLinearRegressor
    SHIFT_RF = ShiftedRFRegressor
    SHIFT_KNN = ShiftedKNNRegressor
    SHIFT_DT = ShiftedDecissionTreeRegressor
    SHIFT_SVR = ShiftedSupportVectorRegressor
    SHIFT_SGD = ShiftedStochasticGradientDescentRegressor
    SHIFT_MLP = ShiftedMultiLayerPerceptronRegressor
    CHAIN_LR = ChainedLinearRegressor
    CHAIN_RF = ChainedRFRegressor
    CHAIN_KNN = ChainedKNNRegressor
    CHAIN_DT = ChainedDecissionTreeRegressor
    CHAIN_SVR = ChainedSupportVectorRegressor
    CHAIN_SGD = ChainedStochasticGradientDescentRegressor
    CHAIN_MLP = ChainedMultiLayerPerceptronRegressor
    MULTI_LR = MultioutputLinearRegressor
    MULTI_RF = MultioutputRFRegressor
    MULTI_KNN = MultioutputKNNRegressor
    MULTI_DT = MultioutputDecissionTreeRegressor
    MULTI_SVR = MultioutputSVMRegressor
    MULTI_SGD = MultioutputSGDRegressor
    MULTI_MLP = MultioutputMLPRegressor
