"""Training related functions"""
import json
import logging
import os
import shutil
import string
import sys
from argparse import Namespace
from typing import Any, Dict, List, Tuple

import pandas as pd  # type: ignore
import scikit_posthocs as sp  # type: ignore
from matplotlib import pyplot as plt  # type: ignore
from scipy.stats import friedmanchisquare  # type: ignore

from .. import panda_utils as pdu
from ..errors import ErrorType
from ..load_sets import get_train_datasets
from ..models import BasicModel, ModelGroups, Models

logger = logging.getLogger("logger")


def select_hyperparameters(
    series: Dict[str, List[str]], args: Namespace
) -> None:
    """generate and fit hyperparameters"""
    base_path = "model-training"
    # Clean training directory
    for check_path in [
        os.path.join(base_path, "errors"),
        os.path.join(base_path, "post-hoc"),
        os.path.join(base_path, "test"),
        os.path.join(base_path, "train"),
    ]:
        if os.path.exists(check_path):
            shutil.rmtree(check_path)
    # Generate models
    train_to_fit_hyperparameters(
        args.models_training,
        get_train_datasets(
            series,
            args.training_percentage_hyperparams,
            args.num_features,
            args.num_predictions,
            args.train_test_ratio_hyperparams,
        ),
        ErrorType[args.error_type_models],
        base_path,
    )


def get_best_trained_model(models: List[str], base_path: str) -> BasicModel:
    """Get best trained model"""
    # Find best model
    _models: List[Tuple[float, str, Dict[str, Any]]] = []
    for model_name in models:
        filename = f"{base_path}/hyperparameters/{model_name}.json"
        error_value, hyperparameters = load_error_and_hyperparameters(filename)
        _models.append((error_value, model_name, hyperparameters))
    _models.sort(key=lambda x: x[0])
    error_value, model_name, hyperparameters = _models[0]
    logger.debug(
        "Best model: %s => %f.6 %r", model_name, error_value, hyperparameters
    )
    # Create model and train it
    model: BasicModel = Models[model_name].value(hyperparameters, base_path)
    return model


def load_error_and_hyperparameters(
    filename: str,
) -> Tuple[float, Dict[str, Any]]:
    """load error and hyperparameters"""
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


def train_to_fit_hyperparameters(
    model_list: List[str],
    datasets: Tuple[pd.DataFrame, pd.DataFrame],
    error: ErrorType,
    base_path: str,
) -> None:
    """Train dataset to obtain the best hyperparameter set"""
    train_features, train_output = datasets
    # Train models
    models_dict = {}
    for model_name in model_list:
        models_dict[model_name] = ModelGroups[model_name].value(
            model_name,
            (train_features, train_output),
            error,
            base_path,
        )
    del train_features
    del train_output
    del datasets
    for model in models_dict.values():
        model.fit()
    hyperparams_list = [
        h for x in models_dict.values() for h in x.hyper_params.keys()
    ]
    logger.debug("Hyperparam list: %r", hyperparams_list)
    friedman_test_with_post_hoc(hyperparams_list, base_path)


def friedman_test_with_post_hoc(
    models: List[str],
    base_path: str,
) -> None:
    """Performa a friedman test with post hoc"""
    # Sanity check
    if len(models) < 3:
        logger.debug("Cannot perform friedman test => %d < 3", len(models))
        return
    # Generate dataset
    sum_errors: List[pd.DataFrame] = []
    for model in models:
        filename = os.path.join(base_path, "sum_errors", f"{model}.csv")
        sum_errors.append(
            pdu.read_csv(filename).set_index(["dataset", "timeseries"])
        )
    friedman_df = pd.concat(sum_errors, axis=1)
    columns = friedman_df.columns
    # Calculate friedman chi-square
    logger.debug("Friedman\n%r", friedman_df)
    f_test = friedmanchisquare(*[friedman_df[k] for k in columns])
    logger.debug("Friedmanchisquare: %r", f_test)
    # Calculate post-hoc if p_value < 0.05
    if f_test.pvalue < 0.05:
        new_columns = dict(zip(string.ascii_uppercase, columns))
        friedman_df.columns = new_columns.keys()
        post_hoc = sp.posthoc_nemenyi_friedman(
            friedman_df.reset_index().iloc[:, 2:]
        )
        heatmap_args = {
            "linewidths": 0.25,
            "linecolor": "0.5",
            "clip_on": False,
            "square": True,
            "cbar_ax_bbox": [0.80, 0.35, 0.04, 0.3],
        }
        sp.sign_plot(post_hoc, **heatmap_args)
        filename = os.path.join(base_path, "post-hoc", "posthoc.png")
        logger.debug("Saving diagram to: %s", filename)
        plt.savefig(filename)
        with open(
            os.path.join(base_path, "post-hoc", "legend.csv"),
            "w",
            encoding="utf8",
        ) as f_legend:
            f_legend.write(json.dumps(new_columns, indent=4))
