"""Models"""
from enum import Enum

from .abstract_model import BasicModel
from .knn_regression import KNNRegressor, MultioutputKNNRegressor
from .linear_regression import LinearRegressor, MultioutputLinearRegressor
from .random_forest_regression import MultioutputRFRegressor, RFRegressor
from .svm_regression import SupportVectorRegressor
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
    MULTI_LR = MultioutputLinearRegressor
    MULTI_RF = MultioutputRFRegressor
    MULTI_KNN = MultioutputKNNRegressor
    MULTI_DT = MultioutputDecissionTreeRegressor


__all__ = [
    "BasicModel",
    "Models",
]
