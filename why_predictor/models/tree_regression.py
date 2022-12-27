"""Decission Tree Regression model"""
import logging

import pandas as pd  # type: ignore
from sklearn.multioutput import RegressorChain  # type: ignore
from sklearn.tree import DecisionTreeRegressor  # type: ignore

from .abstract_model import NUM_HEADERS, MultioutputModel, ShiftedModel

logger = logging.getLogger("logger")


class ShiftedDecissionTreeRegressor(ShiftedModel):
    """Shifted Decission Tree Regression Class"""

    name = "Shifted Decission Tree Regression"
    short_name = "SHIFT_DT"

    def generate_model(
        self, features: pd.DataFrame, output: pd.DataFrame
    ) -> None:
        """Generate model"""
        shifted_dt_model = DecisionTreeRegressor(**self.hyperparams)
        self._model = shifted_dt_model.fit(
            features.drop(["dataset", "timeseries"], axis=1),
            output.iloc[:, NUM_HEADERS],
        )


class ChainedDecissionTreeRegressor(MultioutputModel):
    """ChainedDecission Tree Regression Class"""

    name = "Chained Decission Tree Regression"
    short_name = "CHAIN_DT"

    def generate_model(
        self, features: pd.DataFrame, output: pd.DataFrame
    ) -> None:
        """Generate model"""
        chained_dt_model = RegressorChain(
            DecisionTreeRegressor(**self.hyperparams)
        )
        self._model = chained_dt_model.fit(
            features.drop(["dataset", "timeseries"], axis=1),
            output.drop(["dataset", "timeseries"], axis=1),
        )


class MultioutputDecissionTreeRegressor(MultioutputModel):
    """Decission Tree Regression Class"""

    name = "Multioutput Decission Tree Regression"
    short_name = "MULTI_DT"

    def generate_model(
        self, features: pd.DataFrame, output: pd.DataFrame
    ) -> None:
        """Generate model"""
        multi_dt_model = DecisionTreeRegressor(**self.hyperparams)
        self._model = multi_dt_model.fit(
            features.drop(["dataset", "timeseries"], axis=1),
            output.drop(["dataset", "timeseries"], axis=1),
        )
