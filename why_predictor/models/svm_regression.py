"""Support Vector Regression model"""
import logging

import pandas as pd  # type: ignore
from sklearn.multioutput import (  # type: ignore
    MultiOutputRegressor,
    RegressorChain,
)
from sklearn.svm import LinearSVR  # type: ignore

from .abstract_model import NUM_HEADERS, MultioutputModel, ShiftedModel

logger = logging.getLogger("logger")


class ShiftedSupportVectorRegressor(ShiftedModel):
    """Shifted Support Vector Regressor"""

    name = "Shifted Support Vector Regression"
    short_name = "SHIFT_SVR"

    def generate_model(
        self, features: pd.DataFrame, output: pd.DataFrame
    ) -> None:
        """Generate model"""
        # We train with only the column for the first hour
        shifted_svr_model = LinearSVR(**self.hyperparams, max_iter=10000)
        self._model = shifted_svr_model.fit(
            features.drop(["dataset", "timeseries"], axis=1),
            output.iloc[:, NUM_HEADERS],
        )


class ChainedSupportVectorRegressor(MultioutputModel):
    """Chained Support Vector Regressor"""

    name = "Chained Support Vector Regression"
    short_name = "CHAIN_SVR"

    def generate_model(
        self, features: pd.DataFrame, output: pd.DataFrame
    ) -> None:
        """Generate model"""
        # We train with only the column for the first hour
        chained_svr_model = RegressorChain(
            LinearSVR(**self.hyperparams, max_iter=10000)
        )
        self._model = chained_svr_model.fit(
            features.drop(["dataset", "timeseries"], axis=1),
            output.drop(["dataset", "timeseries"], axis=1),
        )


class MultioutputSVMRegressor(MultioutputModel):
    """Multioutput Support Vector Regressor"""

    name = "Multioutput Support Vector Regression"
    short_name = "MULTI_SVR"

    def generate_model(
        self, features: pd.DataFrame, output: pd.DataFrame
    ) -> None:
        """Generate model"""
        multi_svr_model = MultiOutputRegressor(
            LinearSVR(**self.hyperparams, max_iter=10000)
        )
        self._model = multi_svr_model.fit(
            features.drop(["dataset", "timeseries"], axis=1),
            output.drop(["dataset", "timeseries"], axis=1),
        )
