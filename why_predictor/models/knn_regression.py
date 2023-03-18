"""Linear Regression model"""
import logging

import pandas as pd  # type: ignore
from sklearn.multioutput import RegressorChain  # type: ignore
from sklearn.neighbors import KNeighborsRegressor  # type: ignore

from .. import config
from .abstract_model import NUM_HEADERS, MultioutputModel, ShiftedModel

logger = logging.getLogger("logger")


class ShiftedKNNRegressor(ShiftedModel):
    """Shifted Chained KNN Regressor"""

    name = "Shifted KNN Regression"
    short_name = "SHIFT_KNN"

    def generate_model(
        self, features: pd.DataFrame, output: pd.DataFrame
    ) -> None:
        """Generate model"""
        # We train with only the column for the first hour
        logger.debug(
            "Training %s model (%s)...", self.short_name, self.hyperparams
        )
        shifted_knn_model = KNeighborsRegressor(
            **self.hyperparams, n_jobs=config.NJOBS
        )
        self._model = shifted_knn_model.fit(
            features.drop(["dataset", "timeseries"], axis=1),
            output.iloc[:, NUM_HEADERS],
        )
        logger.debug("%s model trained.", self.short_name)


class ChainedKNNRegressor(MultioutputModel):
    """Chained KNN Regressor"""

    name = "Chained KNN Regression"
    short_name = "CHAIN_KNN"

    def generate_model(
        self, features: pd.DataFrame, output: pd.DataFrame
    ) -> None:
        """Generate model"""
        logger.debug(
            "Training %s model (%s)...", self.short_name, self.hyperparams
        )
        chained_knn_model = RegressorChain(
            KNeighborsRegressor(**self.hyperparams, n_jobs=config.NJOBS)
        )
        self._model = chained_knn_model.fit(
            features.drop(["dataset", "timeseries"], axis=1),
            output.drop(["dataset", "timeseries"], axis=1),
        )
        logger.debug("%s model trained.", self.short_name)


class MultioutputKNNRegressor(MultioutputModel):
    """Multioutput KNN Regressor"""

    name = "Multioutput KNN Regression"
    short_name = "MULTI_KNN"

    def generate_model(
        self, features: pd.DataFrame, output: pd.DataFrame
    ) -> None:
        """Generate model"""
        logger.debug(
            "Training %s model (%s)...", self.short_name, self.hyperparams
        )
        multi_knn_model = KNeighborsRegressor(
            **self.hyperparams, n_jobs=config.NJOBS
        )
        self._model = multi_knn_model.fit(
            features.drop(["dataset", "timeseries"], axis=1),
            output.drop(["dataset", "timeseries"], axis=1),
        )
        logger.debug("%s model trained.", self.short_name)
