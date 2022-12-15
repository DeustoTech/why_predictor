"""Decission Tree Regression model"""
import logging
from typing import Any, Dict, List, Literal, Optional, TypedDict, Union, cast

import pandas as pd  # type: ignore
from sklearn.multioutput import RegressorChain  # type: ignore
from sklearn.tree import DecisionTreeRegressor  # type: ignore

from ..errors import ErrorType
from .abstract_model import BasicModel, MultioutputModel, ShiftedModel
from .utils import generate_hyperparams_from_keys, sanitize_params

logger = logging.getLogger("logger")


DTHyperParamKeys = Literal[
    "max_depth",
    # "criterion",
    # "splitter",
    # "min_samples_split",
    # "min_samples_leaf",
    # "max_features",
    # "max_leaf_nodes",
    # "min_weight_fraction_leaf",
    # "min_impurity_decrease",
    # "ccp_alpha",
]


class DTHyperParams(TypedDict):
    """Decission Tree HyperParams type"""

    max_depth: List[Union[None, int]]
    # criterion: List[str]
    # splitter: List[str]
    # min_samples_split: List[int]
    # min_samples_leaf: List[float]
    # max_features: List[Union[int, float, str]]
    # max_leaf_nodes: List[Union[None, int]]
    # min_weight_fraction_leaf: List[Union[int, float]]
    # min_impurity_decrease: List[float]
    # ccp_alpha: List[float]


class DecissionTreeRegressionModel(BasicModel):
    """Decission Tree Regression Class"""

    params: DTHyperParams = {
        "max_depth": [None, 10, 15],
        # "criterion": [
        #     "squared_error",
        #     "absolute_error",
        #     "friedman_mse",
        #     "poisson",
        # ],
        # "splitter": ["best", "random"],
        # "min_samples_split": [2, 3, 4, 5],
        # "min_samples_leaf": [1, 2, 3, 4, 5],
        # "min_weight_fraction_leaf": [0.0, 0.5],
        # "max_features": [1.0, "sqrt", "log2"],
        # "max_leaf_nodes": [None, 5, 10],
        # "min_impurity_decrease": [0.0, 0.5, 1, 1.5],
        # "ccp_alpha": [0.0, 0.5, 1, 1.5],
        # "ccp_alpha": [0.0, 0.5],
    }

    def __init__(
        self,
        train_features: pd.DataFrame,
        train_output: pd.DataFrame,
        error_type: ErrorType,
        params: Optional[DTHyperParams] = None,
    ):
        self.__params = sanitize_params(params) if params else self.params
        super().__init__(train_features, train_output, error_type)

    def generate_hyperparams(self) -> None:
        """Generate hyperparams"""
        keys: List[DTHyperParamKeys] = cast(
            List[DTHyperParamKeys], list(self.__params.keys())
        )
        hyperparams = generate_hyperparams_from_keys(self.__params, {}, keys)
        self.generate_hyperparams_objects(hyperparams)


class ShiftedDecissionTreeRegressor(
    DecissionTreeRegressionModel, ShiftedModel
):
    """Shifted Decission Tree Regression Class"""

    name = "Shifted Decission Tree Regression"
    short_name = "SHIFT_DT"

    def generate_model(self, hyper_params: Dict[str, Any]) -> Any:
        """Generate model"""
        model = DecisionTreeRegressor(**hyper_params)
        dt_model = model.fit(
            self.train_features.drop("timeseries", axis=1),
            self.train_output.iloc[:, 1],
        )
        return dt_model


class ChainedDecissionTreeRegressor(
    DecissionTreeRegressionModel, MultioutputModel
):
    """ChainedDecission Tree Regression Class"""

    name = "Chained Decission Tree Regression"
    short_name = "CHAIN_DT"

    def generate_model(self, hyper_params: Dict[str, Any]) -> Any:
        """Generate model"""
        model = RegressorChain(DecisionTreeRegressor(**hyper_params))
        chained_dt_model = model.fit(
            self.train_features.drop("timeseries", axis=1),
            self.train_output.drop("timeseries", axis=1),
        )
        return chained_dt_model


class MultioutputDecissionTreeRegressor(
    DecissionTreeRegressionModel, MultioutputModel
):
    """Decission Tree Regression Class"""

    name = "Multioutput Decission Tree Regression"
    short_name = "MULTI_DT"

    def generate_model(self, hyper_params: Dict[str, Any]) -> Any:
        """Generate model"""
        model = DecisionTreeRegressor(**hyper_params)
        multi_dt_model = model.fit(
            self.train_features.drop("timeseries", axis=1),
            self.train_output.drop("timeseries", axis=1),
        )
        return multi_dt_model
