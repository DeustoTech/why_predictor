"""Linear Regression model"""
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict

import pandas as pd  # type: ignore

from ..errors import ErrorType

logger = logging.getLogger("logger")


class BasicModel(ABC):
    """Models must define these methods and properties"""

    name: str

    def __init__(
        self,
        train_features: pd.DataFrame,
        train_output: pd.DataFrame,
    ):
        self.train_features = train_features
        self.train_output = train_output
        self.model = self.generate_model()
        self.errors: Dict[str, Any] = {}
        self.predictions = Any

    @abstractmethod
    def fit(
        self, test_features: pd.DataFrame, test_output: pd.DataFrame
    ) -> None:
        """Generate predictions"""

    @abstractmethod
    def generate_model(self) -> Any:
        """Generate model"""


class ChainedModel(BasicModel):
    """Chained Basic Model"""

    def fit(
        self, test_features: pd.DataFrame, test_output: pd.DataFrame
    ) -> None:
        """Generate predictions"""
        logger.debug("Calculating %s...", self.name)
        predictions = pd.DataFrame(self.model.predict(test_features))
        for i in range(1, test_output.shape[1]):
            features = pd.concat(
                [test_features.iloc[:, i:], predictions], axis=1
            )
            features = features.set_axis(
                [f"col{i}" for i in range(1, features.shape[1] + 1)], axis=1
            )
            predictions = pd.concat(
                [predictions, pd.Series(self.model.predict(features))], axis=1
            )
        self.predictions = predictions.set_axis(test_output.columns, axis=1)
        # Calculate errors
        for error in ErrorType:
            error_metric = error.value(
                test_output, self.predictions, self.train_features
            )
            self.errors[error.name] = error_metric
            logger.info("%s %s:\n%r", error.name, self.name, error_metric)


class MultioutputModel(BasicModel):
    """Multioutput Basic Model"""

    def fit(
        self, test_features: pd.DataFrame, test_output: pd.DataFrame
    ) -> None:
        """Generate predictions"""
        logger.debug("Calculating %s...", self.name)
        predictions = pd.DataFrame(self.model.predict(test_features))
        self.predictions = predictions.set_axis(test_output.columns, axis=1)
        logger.debug(
            "Accuracy: %r", self.model.score(test_features, test_output)
        )
        # Calculate errors
        for error in ErrorType:
            error_metric = error.value(
                test_output, self.predictions, self.train_features
            )
            self.errors[error.name] = error_metric
            logger.info("%s %s:\n%r", error.name, self.name, error_metric)
