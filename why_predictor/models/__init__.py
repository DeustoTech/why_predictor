"""Models"""
from enum import Enum

from .knn_regression import KNNRegressionModel
from .linear_regression import LinearRegressionModel
from .multioutput_knn_regression import MultioutputKNNRegressionModel
from .multioutput_linear_regression import MultioutputLinearRegressionModel
from .multioutput_random_forest_regression import (
    MultioutputRandomForestRegressionModel,
)
from .multioutput_tree_regression import (
    MultioutputDecissionTreeRegressionModel,
)
from .random_forest_regression import RandomForestRegressionModel
from .tree_regression import DecissionTreeRegressionModel


class Models(Enum):
    """Enum with Models"""

    LR = LinearRegressionModel
    RF = RandomForestRegressionModel
    KNN = KNNRegressionModel
    DT = DecissionTreeRegressionModel
    MULTI_LR = MultioutputLinearRegressionModel
    MULTI_RF = MultioutputRandomForestRegressionModel
    MULTI_KNN = MultioutputKNNRegressionModel
    MULTI_DT = MultioutputDecissionTreeRegressionModel
