"""Errors"""
from enum import Enum
from functools import partial
from typing import Any

import numpy as np
import pandas as pd  # type: ignore


def calculate_mape2(
    output: pd.DataFrame, predictions: pd.DataFrame, mean: float
) -> Any:
    """Calculate MAPE2"""
    # Sanity Check
    if output.shape != predictions.shape:
        raise ValueError("Different dimensions out - pred")
    output2 = output[output != 0].fillna(mean)
    # We cast the values to 'float32' to prevent unsigned integers
    return np.absolute(
        (output.astype("float32") - predictions.astype("float32")) / output2
    )


def calculate_mape(
    output: pd.DataFrame, predictions: pd.DataFrame, _: float = 0.0
) -> Any:
    """Calculate MAPE"""
    # Sanity Check
    if output.shape != predictions.shape:
        raise ValueError("Different dimensions out - pred")
    # We cast the values to 'float32' to prevent unsigned integers
    return np.absolute(
        (output.astype("float32") - predictions.astype("float32")) / output
    )


def calculate_mae(
    output: pd.DataFrame, predictions: pd.DataFrame, _: float = 0.0
) -> Any:
    """Calculate MAE"""
    # Sanity Check
    if output.shape != predictions.shape:
        raise ValueError("Different dimensions out - pred")
    # We cast the values to 'float32' to prevent unsigned integers
    return np.absolute(
        (output.astype("float32") - predictions.astype("float32"))
    )


def calculate_rmse(
    output: pd.DataFrame, predictions: pd.DataFrame, _: float = 0.0
) -> Any:
    """Calculate RMSE"""
    # Sanity Check
    if output.shape != predictions.shape:
        raise ValueError("Different dimensions out - pred")
    # We cast the values to 'float32' to prevent unsigned integers
    return np.square(output.astype("float32") - predictions.astype("float32"))


def calculate_smape(
    output: pd.DataFrame, predictions: pd.DataFrame, _: float = 0.0
) -> Any:
    """Calculate SMAPE"""
    # Sanity Check
    if output.shape != predictions.shape:
        raise ValueError("Different dimensions out - pred")
    # We cast the values to 'float32' to prevent unsigned integers
    return np.absolute(
        predictions.astype("float32") - output.astype("float32")
    ) / ((np.absolute(output) + np.absolute(predictions)) / 2)


class ErrorType(Enum):
    """Enum with error calculators"""

    MAPE2 = partial(calculate_mape2)
    MAPE = partial(calculate_mape)
    MAE = partial(calculate_mae)
    RMSE = partial(calculate_rmse)
    SMAPE = partial(calculate_smape)
