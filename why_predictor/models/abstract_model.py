"""Linear Regression model"""
import json
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, TypedDict

import pandas as pd  # type: ignore

from ..errors import ErrorType

logger = logging.getLogger("logger")


class HyperParams(TypedDict):
    """HyperParams type"""

    name: str
    params: Dict[str, Any]
    model: Any
    errors: pd.DataFrame
    median: float


class BasicModel(ABC):
    """Models must define these methods and properties"""

    name: str

    def __init__(
        self,
        train_features: pd.DataFrame,
        train_output: pd.DataFrame,
        error_type: ErrorType,
    ):
        logger.debug("Generating %s...", self.name)
        self.train_features = train_features
        self.train_output = train_output
        self.error_type = error_type
        self.hyper_params: Dict[str, HyperParams] = {}
        self.predictions = Any
        self.fitted: HyperParams
        self.generate_hyperparams()

    @abstractmethod
    def generate_hyperparams(self) -> None:
        """Generate hyperparams"""

    def generate_hyperparams_objects(self, hyperparams: List[Any]) -> None:
        """Generate HyperParams Object"""
        logger.debug(
            "Generating hyper params for %s (%d)...",
            self.name,
            len(hyperparams),
        )
        for hyperparam_set in hyperparams:
            name = json.dumps(hyperparam_set)
            self.hyper_params[name] = {
                "name": name,
                "params": hyperparam_set,
                "model": self.generate_model(hyperparam_set),
                "errors": pd.DataFrame,
                "median": 0.0,
            }
            # TODO friendman with post-hoc
            # https://stats.stackexchange.com/questions/467467/how-to-do-friedman-test-and-post-hoc-test-on-python

    @abstractmethod
    def generate_model(self, hyper_params: Dict[str, Any]) -> Any:
        """Generate model"""

    @abstractmethod
    def calculate_errors(
        self,
        hyperparams_set: str,
        test_features: pd.DataFrame,
        test_output: pd.DataFrame,
    ) -> None:
        """Calculate errors for a hyper param set"""

    def fit(
        self, test_features: pd.DataFrame, test_output: pd.DataFrame
    ) -> None:
        """Generate predictions"""
        logger.debug("Calculating %s...", self.name)
        for hyperparams_set in self.hyper_params:
            self.calculate_errors(hyperparams_set, test_features, test_output)
        self.fitted = sorted(
            self.hyper_params.values(), key=lambda x: x["median"]
        )[0]
        logger.debug(
            "Best hyper params set: %s => %f",
            self.fitted["name"],
            self.fitted["median"],
        )


class ChainedModel(BasicModel):
    """Chained Basic Model"""

    def calculate_errors(
        self,
        hyperparams_set: str,
        test_features: pd.DataFrame,
        test_output: pd.DataFrame,
    ) -> None:
        """Calculate errors for a hyper param set"""
        hyperparams = self.hyper_params[hyperparams_set]
        logger.debug("Hyper params set: %s", hyperparams["name"])
        predictions = pd.DataFrame(hyperparams["model"].predict(test_features))
        for i in range(1, test_output.shape[1]):
            # We generate a new features vector, removing first columns and
            # adding the already predicted values as features
            # 1 2 [3 4 5 6 7 ... 70 71 72 P1 P2]
            features = pd.concat(
                [test_features.iloc[:, i:], predictions], axis=1
            )
            features = features.set_axis(
                [f"col{i}" for i in range(1, features.shape[1] + 1)], axis=1
            )
            predictions = pd.concat(
                [
                    predictions,
                    pd.Series(hyperparams["model"].predict(features)),
                ],
                axis=1,
            )
        self.predictions = predictions.set_axis(test_output.columns, axis=1)
        # Calculate errors
        error_metric = self.error_type.value(
            test_output, self.predictions, self.train_features
        )
        hyperparams["errors"] = error_metric
        hyperparams["median"] = error_metric.stack().median()
        logger.info(
            "%s %s: %r", self.error_type.name, self.name, hyperparams["median"]
        )
        logger.debug(error_metric)


class MultioutputModel(BasicModel):
    """Multioutput Basic Model"""

    def calculate_errors(
        self,
        hyperparams_set: str,
        test_features: pd.DataFrame,
        test_output: pd.DataFrame,
    ) -> None:
        """Calculate errors for a hyper param set"""
        hyperparams = self.hyper_params[hyperparams_set]
        logger.debug("Hyper params set: %s", hyperparams["name"])
        predictions = pd.DataFrame(hyperparams["model"].predict(test_features))
        self.predictions = predictions.set_axis(test_output.columns, axis=1)
        logger.debug(
            "Accuracy: %r",
            hyperparams["model"].score(test_features, test_output),
        )
        # Calculate errors
        error_metric = self.error_type.value(
            test_output, self.predictions, self.train_features
        )
        hyperparams["errors"] = error_metric
        hyperparams["median"] = error_metric.stack().median()
        logger.info("%s %s: %r", self.error_type.name, self.name, error_metric)
