"""FFORMA related functions"""
import logging
import math
import os
import random
import shutil
from argparse import Namespace
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd  # type: ignore

from .. import panda_utils as pdu
from ..errors import ErrorType
from ..load_sets import get_train_datasets
from ..models import BasicModel, Models
from .models import (
    get_best_trained_model,
    load_error_and_hyperparameters,
    train_to_fit_hyperparameters,
)

logger = logging.getLogger("logger")


def final_fforma_prediction(
    series: Dict[str, List[str]],
    training_percentage_fforma: float,
    train_test_ratio_fforma: float,
    args: Namespace,
) -> None:
    """final FFORMA prediction"""
    # Get FFORMA datasets
    (
        train_features,
        train_output,
        models,
    ) = get_datasets_and_dict_trained_models(
        args.models_training,
        get_train_datasets(
            series,
            training_percentage_fforma,
            args.num_features,
            args.num_predictions,
            train_test_ratio_fforma,
        ),
        ErrorType[args.error_type_fforma],
        train_test_ratio_fforma,
        "model-training",
    )
    # Get FFORMA Model
    get_best_trained_model(
        args.models_fforma,
        base_path="fforma-training",
    )
    # Delete train dataset to free memory
    del train_features
    del train_output
    # Get FFORMA output
    _generate_fforma_final_output(
        models,
        pdu.read_csv(
            "fforma-training/test/fforma/output/dataset.csv.gz"
        ).set_index(["dataset", "timeseries"]),
        ErrorType[args.error_type_fforma],
    )


def _generate_fforma_final_output(
    models: Dict[str, BasicModel],
    fforma_test_output: pd.DataFrame,
    error: ErrorType,
) -> None:
    error_filename = "fforma-training/evaluation_errors.csv.gz"
    if os.path.exists(error_filename):
        os.remove(error_filename)
    logger.debug("Fforma test_output:\n%r", fforma_test_output)
    for dataset in fforma_test_output.index:
        test_features = pdu.read_csv(
            f"model-training/test/{dataset[0]}/features/{dataset[1]}.csv.gz"
        )
        test_output = pdu.read_csv(
            f"model-training/test/{dataset[0]}/output/{dataset[1]}.csv.gz"
        )
        final_output: pd.DataFrame = None
        for model in models.values():
            value = model.make_predictions(test_features, test_output)
            value = value * fforma_test_output[model.short_name][dataset]
            if final_output is not None:
                final_output += value
            else:
                final_output = value
            model.clear_model()
        fforma_test_output["sum"] = fforma_test_output.sum(axis=1)
        final_output = final_output / fforma_test_output["sum"][dataset]
        logger.debug("Final output\n%r", final_output)
        error.value(
            test_output.iloc[:, 2:],
            final_output,
            test_features.iloc[:, 2:].stack().median(),
        ).to_csv(error_filename, mode="a", index=False, header=False)
    del fforma_test_output
    logger.info(
        "Final evalution median error: %r",
        pdu.read_csv(error_filename, header=None).stack().median(),
    )


def train_fforma(
    series: Dict[str, List[str]],
    training_percentage_fforma: float,
    train_test_ratio_fforma: float,
    args: Namespace,
) -> None:
    """Train FFORMA"""
    fforma_path = "fforma-training"
    # Clean training directory
    for check_path in [
        os.path.join(fforma_path, "errors"),
        os.path.join(fforma_path, "post-hoc"),
        os.path.join(fforma_path, "test"),
    ]:
        if os.path.exists(check_path):
            shutil.rmtree(check_path)
    if os.path.exists("model-training/test"):
        shutil.rmtree("model-training/test")
    # Train models
    train_to_fit_hyperparameters(
        args.models_fforma,
        get_fforma_train_test_datasets(
            series, training_percentage_fforma, train_test_ratio_fforma, args
        ),
        ErrorType[args.error_type_models],
        fforma_path,
    )


def get_fforma_train_test_datasets(
    series: Dict[str, List[str]],
    training_percentage_fforma: float,
    train_test_ratio_fforma: float,
    args: Namespace,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Get train and test fforma datasets"""
    # Calculate models
    train_features, train_output, _ = get_datasets_and_dict_trained_models(
        args.models_training,
        get_train_datasets(
            series,
            training_percentage_fforma,
            args.num_features,
            args.num_predictions,
            train_test_ratio_fforma,
        ),
        ErrorType[args.error_type_fforma],
        train_test_ratio_fforma,
        "model-training",
    )
    return train_features, train_output


def get_datasets_and_dict_trained_models(
    models: List[str],
    datasets: Tuple[pd.DataFrame, pd.DataFrame],
    error: ErrorType,
    train_test_ratio_fforma: float,
    models_base_path: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, BasicModel]]:
    """Train models"""
    train_features, train_output = datasets
    median_value = np.nanmean(train_features.iloc[:, 2:])
    # Generating models with best hyperparamer set
    models_dict = _get_best_models_dict(
        models, train_features, train_output, models_base_path
    )
    # group by 'dataset' and get timeseries list
    dataset_grouped_list = [
        (x, [t[0] for t in v.groupby(["dataset", "timeseries"])])
        for x, v in train_output[["dataset", "timeseries"]].groupby("dataset")
    ]
    # Delete train dataset to free memory
    del train_features
    del train_output
    # create two datasets one for train and another for test
    train_datasets, test_datasets = _generate_train_test_fforma_datasets(
        dataset_grouped_list, train_test_ratio_fforma
    )
    del dataset_grouped_list
    # Load context
    train_features = pdu.read_csv("context.csv")
    train_features.timeseries = train_features.timeseries.astype("string")
    train_features.set_index(["dataset", "timeseries"], inplace=True)
    # Generate path
    fforma_base_path = "fforma-training/test/fforma/"
    if not os.path.exists(fforma_base_path):
        os.makedirs(os.path.join(fforma_base_path, "features"))
        os.makedirs(os.path.join(fforma_base_path, "output"))
    # Generate test features and save it to file
    train_features[
        train_features.index.isin(test_datasets)
    ].reset_index().to_csv(
        os.path.join(fforma_base_path, "features", "dataset.csv.gz"),
        index=False,
    )
    # Generate train features dataset
    train_features = train_features[
        train_features.index.isin(train_datasets)
    ].reset_index()
    # For test dataset
    _generate_fforma_output_dataset(
        # [pdu.DataFrame(test_datasets, columns=["dataset", "timeseries"])],
        test_datasets,
        models_dict,
        error,
        median_value,
        fforma_base_path,
    )
    # For train dataset
    train_output = _generate_fforma_output_dataset(
        # [pdu.DataFrame(train_datasets, columns=["dataset", "timeseries"])],
        train_datasets,
        models_dict,
        error,
        median_value,
    )
    return (
        train_features,
        train_output,
        models_dict,
    )


def _get_best_models_dict(
    models: List[str],
    train_features: pd.DataFrame,
    train_output: pd.DataFrame,
    models_base_path: str,
) -> Dict[str, BasicModel]:
    logger.debug("Generating best hyperparameter-set models...")
    models_dict: Dict[str, BasicModel] = {}
    for model_name in models:
        filename = f"{models_base_path}/hyperparameters/{model_name}.json"
        _, hyperparameters = load_error_and_hyperparameters(filename)
        model = Models[model_name].value(hyperparameters, models_base_path)
        model.generate_model(train_features, train_output)
        model.save_model()
        model.clear_model()
        models_dict[model_name] = model
    logger.debug("Best hyperparameter-set models generated.")
    return models_dict


def _generate_train_test_fforma_datasets(
    test_datasets: List[Tuple[str, List[Tuple[str, str]]]],
    train_test_ratio_fforma: float,
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    logger.debug("Generating FFORMA datasets...")
    train_datasets: List[pd.DataFrame] = []
    for dataset, values in test_datasets:
        train_values: pd.DataFrame = []
        limit = math.ceil(len(values) * train_test_ratio_fforma)
        while len(train_values) < limit:
            train_values.append(values.pop(random.randint(0, len(values) - 1)))
        train_datasets.append((dataset, train_values))
    return (
        [(row[0], str(row[1])) for x in train_datasets for row in x[1]],
        [(row[0], str(row[1])) for x in test_datasets for row in x[1]],
    )


def _generate_fforma_output_dataset(
    datasets: List[Tuple[str, str]],
    models_dict: Dict[str, BasicModel],
    error: ErrorType,
    median_value: float,
    fforma_base_path: Optional[str] = None,
) -> pd.DataFrame:
    dtfs = [pdu.DataFrame(datasets, columns=["dataset", "timeseries"])]
    for model in models_dict.values():
        values = []
        for dataset in datasets:
            values.append(
                model.calculate_timeseries_error(dataset, error, median_value)
            )
        model.clear_model()
        dtfs.append(pdu.DataFrame(values, columns=[model.short_name]))
    result = pdu.concat(dtfs, axis=1)
    if fforma_base_path:
        result.to_csv(
            os.path.join(fforma_base_path, "output", "dataset.csv.gz"),
            index=False,
        )
    return result
