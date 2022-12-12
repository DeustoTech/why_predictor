"""Errors"""
from enum import Enum
from functools import partial
from typing import Any

import numpy as np
import pandas as pd  # type: ignore


def calculate_mape2(
    output: pd.DataFrame, predictions: pd.DataFrame, values: pd.DataFrame
) -> Any:
    """Calculate MAPE2"""
    mean = np.nanmean(values.iloc[:, 1:])
    actual = output.iloc[:, 1:]
    actual2 = actual[actual != 0].fillna(mean)
    predicted = predictions.iloc[:, 1:]
    # return np.absolute((actual - predicted) / actual2).sum() / len(actual)
    return np.absolute((actual - predicted) / actual2)


def calculate_mape(
    output: pd.DataFrame, predictions: pd.DataFrame, _: pd.DataFrame = None
) -> Any:
    """Calculate MAPE"""
    actual = output.iloc[:, 1:]
    predicted = predictions.iloc[:, 1:]
    # return np.absolute((actual - predicted) / actual).sum() / len(actual)
    return np.absolute((actual - predicted) / actual)


def calculate_mae(
    output: pd.DataFrame, predictions: pd.DataFrame, _: pd.DataFrame = None
) -> Any:
    """Calculate MAE"""
    actual = output.iloc[:, 1:]
    predicted = predictions.iloc[:, 1:]
    # return np.absolute((actual - predicted)).sum() / len(actual)
    return np.absolute((actual - predicted))


def calculate_rmse(
    output: pd.DataFrame, predictions: pd.DataFrame, _: pd.DataFrame = None
) -> Any:
    """Calculate RMSE"""
    actual = output.iloc[:, 1:]
    predicted = predictions.iloc[:, 1:]
    # return ((actual - predicted).pow(2).sum() / len(actual)).pow(1.0 / 2)
    return pd.DataFrame(
        ((actual - predicted).pow(2).sum() / len(actual)).pow(1.0 / 2)
    )


def calculate_smape(
    output: pd.DataFrame, predictions: pd.DataFrame, _: pd.DataFrame = None
) -> Any:
    """Calculate SMAPE"""
    actual = output.iloc[:, 1:]
    predicted = predictions.iloc[:, 1:]
    # return (
    #     np.absolute(predicted - actual) / ((actual + predicted) / 2)
    # ).sum() / len(actual)
    return np.absolute(predicted - actual) / ((actual + predicted) / 2)


class ErrorType(Enum):
    """Enum with error calculators"""

    MAPE2 = partial(calculate_mape2)
    MAPE = partial(calculate_mape)
    MAE = partial(calculate_mae)
    RMSE = partial(calculate_rmse)
    SMAPE = partial(calculate_smape)
