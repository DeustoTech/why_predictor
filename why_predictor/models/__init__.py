"""Models"""
from enum import Enum

from .abstract_model import BasicModel
from .model_group import (
    DecissionTreeRegressionModelGroup,
    KNNRegressionModelGroup,
    LinearRegressionModelGroup,
    MLPRegressionModelGroup,
    RandomForestRegressionModelGroup,
    SGDRegressionModelGroup,
    SVMRegressionModelGroup,
)
from .models import Models


class ModelGroups(Enum):
    """Enum with Model-Groups"""

    SHIFT_LR = LinearRegressionModelGroup
    SHIFT_RF = RandomForestRegressionModelGroup
    SHIFT_KNN = KNNRegressionModelGroup
    SHIFT_DT = DecissionTreeRegressionModelGroup
    SHIFT_SVR = SVMRegressionModelGroup
    SHIFT_SGD = SGDRegressionModelGroup
    SHIFT_MLP = MLPRegressionModelGroup
    CHAIN_LR = LinearRegressionModelGroup
    CHAIN_RF = RandomForestRegressionModelGroup
    CHAIN_KNN = KNNRegressionModelGroup
    CHAIN_DT = DecissionTreeRegressionModelGroup
    CHAIN_SVR = SVMRegressionModelGroup
    CHAIN_SGD = SGDRegressionModelGroup
    CHAIN_MLP = MLPRegressionModelGroup
    MULTI_LR = LinearRegressionModelGroup
    MULTI_RF = RandomForestRegressionModelGroup
    MULTI_KNN = KNNRegressionModelGroup
    MULTI_DT = DecissionTreeRegressionModelGroup
    MULTI_SVR = SVMRegressionModelGroup
    MULTI_SGD = SGDRegressionModelGroup
    MULTI_MLP = MLPRegressionModelGroup


__all__ = [
    "BasicModel",
    "Models",
    "ModelGroups",
]
