"""Multi-Layer Perceptron Regression model"""
import logging

import pandas as pd  # type: ignore
from sklearn.multioutput import RegressorChain  # type: ignore
from sklearn.neural_network import MLPRegressor  # type: ignore

from .abstract_model import NUM_HEADERS, MultioutputModel, ShiftedModel

logger = logging.getLogger("logger")


class ShiftedMultiLayerPerceptronRegressor(ShiftedModel):
    """Shifted Multi-layer Preceptron Regressor"""

    name = "Shifted Multi-layer Perceptron Regressor"
    short_name = "SHIFT_MLP"

    def generate_model(
        self, features: pd.DataFrame, output: pd.DataFrame
    ) -> None:
        """Generate model"""
        logger.debug(
            "Training %s model (%s)...", self.short_name, self.hyperparams
        )
        # We train with only the column for the first hour
        shifted_mlp_model = MLPRegressor(**self.hyperparams, max_iter=10000)
        self._model = shifted_mlp_model.fit(
            features.drop(["dataset", "timeseries"], axis=1),
            output.iloc[:, NUM_HEADERS],
        )
        logger.debug("%s model trained.", self.short_name)


class ChainedMultiLayerPerceptronRegressor(MultioutputModel):
    """Chained Multi-layer Perceptron Regressor"""

    name = "Chained Multi-layer Perceptron Regressor"
    short_name = "CHAIN_MLP"

    def generate_model(
        self, features: pd.DataFrame, output: pd.DataFrame
    ) -> None:
        """Generate model"""
        logger.debug(
            "Training %s model (%s)...", self.short_name, self.hyperparams
        )
        # We train with only the column for the first hour
        chained_mlp_model = RegressorChain(
            MLPRegressor(**self.hyperparams, max_iter=10000)
        )
        self._model = chained_mlp_model.fit(
            features.drop(["dataset", "timeseries"], axis=1),
            output.drop(["dataset", "timeseries"], axis=1),
        )
        logger.debug("%s model trained.", self.short_name)


class MultioutputMLPRegressor(MultioutputModel):
    """Multioutput KNN Regressor"""

    name = "Multioutput MLP Regression"
    short_name = "MULTI_MLP"

    def generate_model(
        self, features: pd.DataFrame, output: pd.DataFrame
    ) -> None:
        """Generate model"""
        logger.debug(
            "Training %s model (%s)...", self.short_name, self.hyperparams
        )
        multi_mlp_model = MLPRegressor(**self.hyperparams, max_iter=10000)
        self._model = multi_mlp_model.fit(
            features.drop(["dataset", "timeseries"], axis=1),
            output.drop(["dataset", "timeseries"], axis=1),
        )
        logger.debug("%s model trained.", self.short_name)
