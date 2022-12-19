"""Training related functions"""
import json
import logging
import sys
from argparse import Namespace
from typing import Any, Dict, List, Tuple

from pandas import DataFrame  # type: ignore

from ..errors import ErrorType
from ..load_sets import get_train_and_test_datasets
from ..models import BasicModel, Models

logger = logging.getLogger("logger")


def select_hyperparameters(
    series: Dict[str, List[str]], args: Namespace, base_path: str
) -> None:
    """generate and fit hyperparameters"""
    (
        train_features_set,
        train_output_set,
        test_features_set,
        test_output_set,
    ) = get_train_and_test_datasets(
        series,
        args.training_percentage_hyperparams,
        args.num_features,
        args.num_predictions,
        args.train_test_ratio_hyperparams,
    )
    # Calculate models
    train_models(
        args.models_training,
        (
            train_features_set,
            train_output_set,
            test_features_set,
            test_output_set,
        ),
        ErrorType[args.error_type_models],
        base_path,
    )


def train_models(
    models: List[str],
    datasets: Tuple[DataFrame, DataFrame, DataFrame, DataFrame],
    error: ErrorType,
    base_path: str,
) -> None:
    """Train models"""
    (
        train_features,
        train_output,
        test_features,
        test_output,
    ) = datasets
    for model_name in models:
        Models[model_name].value(train_features, train_output, error).fit(
            test_features, test_output, base_path=base_path
        )


def get_dict_trained_models(
    models: List[str],
    datasets: Tuple[DataFrame, DataFrame, DataFrame, DataFrame],
    error: ErrorType,
    base_path: str,
) -> Dict[str, BasicModel]:
    """Train models"""
    models_dict = {}
    (
        train_features,
        train_output,
        test_features,
        test_output,
    ) = datasets
    for model_name in models:
        filename = f"{base_path}/hyperparameters/{model_name}.json"
        _, hyperparameters = _load_error_and_hyperparameters(filename)
        models_dict[model_name] = Models[model_name].value(
            train_features,
            train_output,
            error,
            hyperparameters,
        )
        models_dict[model_name].fit(
            test_features, test_output, base_path=base_path
        )
    return models_dict


def get_best_trained_model(
    models: List[str],
    datasets: Tuple[DataFrame, DataFrame, DataFrame, DataFrame],
    error: ErrorType,
    base_path: str,
) -> BasicModel:
    """Train models"""
    _models: List[Tuple[float, str, Dict[str, Any]]] = []
    (
        train_features,
        train_output,
        test_features,
        test_output,
    ) = datasets
    for model_name in models:
        filename = f"{base_path}/hyperparameters/{model_name}.json"
        error_value, hyperparameters = _load_error_and_hyperparameters(
            filename
        )
        _models.append((error_value, model_name, hyperparameters))
    _models.sort(key=lambda x: x[0])
    error_value, model_name, hyperparameters = _models[0]
    logger.debug(
        "Best model: %s => %f.6 %r", model_name, error_value, hyperparameters
    )
    model: BasicModel = Models[model_name].value(
        train_features,
        train_output,
        error,
        hyperparameters,
    )
    model.fit(test_features, test_output, base_path=base_path)
    return model


def _load_error_and_hyperparameters(
    filename: str,
) -> Tuple[float, Dict[str, Any]]:
    error: float = 0.0
    hyperparameters: Dict[str, Any]
    try:
        with open(filename, encoding="utf8") as fhyper:
            error_text, hyper_text = fhyper.read().split("|")
            error = float(error_text)
            hyperparameters = json.loads(hyper_text)
    except FileNotFoundError:
        logger.error("File '%s' not found. Aborting...", filename)
        sys.exit(1)
    return (error, hyperparameters)
