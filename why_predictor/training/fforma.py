"""FFORMA related functions"""
import logging
import os
from argparse import Namespace
from typing import Dict, List, Tuple, Union

import pandas as pd  # type: ignore

from ..errors import ErrorType
from ..load_sets import (
    get_train_and_test_datasets,
    split_fforma_in_train_and_test,
)
from ..models import BasicModel
from .models import get_dict_trained_models, train_models

logger = logging.getLogger("logger")


def train_fforma(
    series: Dict[str, List[str]],
    training_percentage_fforma: float,
    train_test_ratio_fforma: float,
    args: Namespace,
) -> None:
    """Train FFORMA"""
    (
        train_features,
        train_output,
        test_features,
        test_output,
    ) = _get_fforma_train_test_datasets(
        series, training_percentage_fforma, train_test_ratio_fforma, args
    )
    train_models(
        args.models_fforma,
        (train_features, train_output, test_features, test_output),
        ErrorType[args.error_type_models],
        base_path="fforma-training",
    )


def _get_fforma_train_test_datasets(
    series: Dict[str, List[str]],
    training_percentage_fforma: float,
    train_test_ratio_fforma: float,
    args: Namespace,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fforma_dataset = generate_fforma_dataset(
        series, training_percentage_fforma, train_test_ratio_fforma, args
    )
    logger.debug("FFORMA dataset:\n%r", fforma_dataset)
    (
        train_features,
        train_output,
        test_features,
        test_output,
    ) = split_fforma_in_train_and_test(
        fforma_dataset,
        args.train_test_ratio_fforma,
        fforma_dataset.shape[1] - len(args.models_training) - 1,
    )
    logger.debug("Features dataset:\n%r", train_features)
    logger.debug("Output dataset:\n%r", train_output)
    return (train_features, train_output, test_features, test_output)


def generate_fforma_dataset(
    file_series: Dict[str, List[str]],
    training_percentage_fforma: float,
    train_test_ratio_fforma: float,
    args: Namespace,
) -> pd.DataFrame:
    """Generate FFORMA ensemble"""
    (
        train_features,
        train_output,
        test_features,
        test_output,
    ) = get_train_and_test_datasets(
        file_series,
        training_percentage_fforma,
        args.num_features,
        args.num_predictions,
        train_test_ratio_fforma,
    )
    # Calculate models
    models_dict = get_dict_trained_models(
        args.models_training,
        (train_features, train_output, test_features, test_output),
        ErrorType[args.error_type_fforma],
        "model-training",
    )
    # Generate FFORMA dataset
    df_fforma = _generate_fforma_dataset(models_dict)
    # Save CSV
    if not os.path.exists("fforma-training"):
        os.makedirs("fforma-training")
    df_fforma.to_csv("fforma-training/fforma.csv", index=False)
    return df_fforma


def _generate_timeseries_dict(
    models_dict: Dict[str, BasicModel]
) -> Dict[str, Dict[str, float]]:
    timeseries_dict: Dict[str, Dict[str, float]] = {}
    for model in models_dict.values():
        for timeseries, error_metric in model.fitted["errors"].groupby(
            "timeseries"
        ):
            if timeseries not in timeseries_dict:
                timeseries_dict[timeseries] = {}
            timeseries_dict[timeseries][model.short_name] = (
                error_metric.drop("timeseries", axis=1).stack().median()
            )
    return timeseries_dict


def _generate_fforma_dataset(
    models_dict: Dict[str, BasicModel]
) -> pd.DataFrame:
    timeseries_dict = _generate_timeseries_dict(models_dict)
    column_names: List[Union[str, float]] = ["timeseries"]
    column_names.extend(["c1", "c2", "c3"])  # TODO to be decided
    column_names.extend([m.short_name for m in models_dict.values()])
    df_fforma = pd.DataFrame(columns=column_names)
    for timeseries_id, timeseries in timeseries_dict.items():
        row_data: List[Union[str, float]] = [timeseries_id, 0.5, 0.5, 0.5]
        row_data.extend(timeseries.values())
        df_fforma = pd.concat(
            [pd.DataFrame([row_data], columns=df_fforma.columns), df_fforma],
            ignore_index=True,
        )
    return df_fforma
