"""Decission Tree Regression model"""
import logging
from typing import Any, Dict, List, Literal, TypedDict, Union, cast

from sklearn.tree import DecisionTreeRegressor  # type: ignore

from .abstract_model import BasicModel, ChainedModel, MultioutputModel

logger = logging.getLogger("logger")


DTHyperParamKeys = Literal[
    "criterion",
    "splitter",
    "min_samples_split",
    "min_samples_leaf",
    "min_weight_fraction_leaf",
    "max_features",
    "max_leaf_nodes",
    "min_impurity_decrease",
    "ccp_alpha",
]


class DTHyperParams(TypedDict):
    """Decission Tree HyperParams type"""

    criterion: List[str]
    splitter: List[str]
    min_samples_split: List[int]
    min_samples_leaf: List[float]
    min_weight_fraction_leaf: List[Union[int, float]]
    max_features: List[Union[int, float, str]]
    max_leaf_nodes: List[Union[None, int]]
    min_impurity_decrease: List[float]
    ccp_alpha: List[float]


class DecissionTreeRegressionModel(BasicModel):
    """Decission Tree Regression Class"""

    params: DTHyperParams = {
        "criterion": [
            "squared_error",
            "friedman_mse",
            "absolute_error",
            "poisson",
        ],
        # "splitter": ["best", "random"],
        "splitter": ["best"],
        # "min_samples_split": [2, 3, 4, 5],
        "min_samples_split": [2],
        # "min_samples_leaf": [1, 2, 3, 4, 5],
        "min_samples_leaf": [1],
        # "min_weight_fraction_leaf": [0.0, 0.5],
        "min_weight_fraction_leaf": [0.0],
        "max_features": [1.0, "sqrt", "log2"],
        "max_leaf_nodes": [None, 5, 10],
        # "min_impurity_decrease": [0.0, 0.5, 1, 1.5],
        "min_impurity_decrease": [0.0],
        # "ccp_alpha": [0.0, 0.5, 1, 1.5],
        "ccp_alpha": [0.0, 0.5],
    }

    def generate_hyperparams(self) -> None:
        """Generate hyperparams"""
        keys: List[DTHyperParamKeys] = cast(
            List[DTHyperParamKeys], list(self.params.keys())
        )
        hyperparams = self.__generate_hyperparams({}, keys)
        self.generate_hyperparams_objects(hyperparams)

    def __generate_hyperparams(
        self,
        current_set: Dict[str, Any],
        hyperparams: List[DTHyperParamKeys],
    ) -> List[Dict[str, Any]]:
        if hyperparams:
            hyperparam = hyperparams.pop(0)
            hyperparam_sets = []
            for value in self.params[hyperparam]:
                my_set = current_set.copy()
                my_set[hyperparam] = value
                hyperparam_sets.extend(
                    self.__generate_hyperparams(my_set, hyperparams[:])
                )
            return hyperparam_sets
        return [current_set]


class DecissionTreeRegressor(DecissionTreeRegressionModel, ChainedModel):
    """Decission Tree Regression Class"""

    name = "Decission Tree Regression"

    def generate_model(self, hyper_params: Dict[str, Any]) -> Any:
        """Generate model"""
        model = DecisionTreeRegressor(**hyper_params)
        dt_model = model.fit(
            self.train_features.drop("timeseries", axis=1),
            self.train_output.iloc[:, 1],
        )
        return dt_model


class MultioutputDecissionTreeRegressor(
    DecissionTreeRegressionModel, MultioutputModel
):
    """Decission Tree Regression Class"""

    name = "Multioutput Decission Tree Regression"

    def generate_model(self, hyper_params: Dict[str, Any]) -> Any:
        """Generate model"""
        model = DecisionTreeRegressor(**hyper_params)
        multi_dt_model = model.fit(
            self.train_features.drop("timeseries", axis=1),
            self.train_output.drop("timeseries", axis=1),
        )
        return multi_dt_model
