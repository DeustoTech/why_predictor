"""Training related functions"""
import json
import logging
import os
import string
from typing import Dict, List, Tuple

import pandas as pd  # type: ignore
import scikit_posthocs as sp  # type: ignore
from matplotlib import pyplot as plt  # type: ignore
from scipy.stats import friedmanchisquare  # type: ignore

from .. import config, loading
from .. import panda_utils as pdu
from ..errors import ErrorType
from ..models import ModelGroups

logger = logging.getLogger("logger")


def select_hyperparameters(series: Dict[str, List[str]]) -> None:
    """generate and fit hyperparameters"""
    # Generate models
    logger.debug("Generating best hyperparameters for models...")
    train_to_fit_hyperparameters(
        config.MODELS_TRAINING,
        loading.datasets.get_train_datasets(
            series,
            config.TRAINING_PERCENTAGE_MODELS,
            (config.NUM_FEATURES, config.NUM_PREDICTIONS),
            config.TRAIN_TEST_RATIO_MODELS,
            config.TRAINING_PATH,
        ),
        ErrorType[config.ERROR_TYPE_MODELS],
        config.TRAINING_PATH,
    )


def select_hyperparameters_initial_path(series: Dict[str, List[str]]) -> None:
    """generate and fit hyperparameters"""
    # Generate models
    logger.debug("Generating best hyperparameters (with specific train)...")
    train_to_fit_hyperparameters(
        config.MODELS_TRAINING,
        loading.datasets.get_train_datasets_from_initial_path(
            series,
            config.TRAINING_PERCENTAGE_MODELS,
            (config.NUM_FEATURES, config.NUM_PREDICTIONS),
            config.INITIAL_TRAINING_PATH,
            config.TRAINING_PATH,
        ),
        ErrorType[config.ERROR_TYPE_MODELS],
        config.TRAINING_PATH,
    )


def train_to_fit_hyperparameters(
    model_list: List[str],
    datasets: Tuple[pd.DataFrame, pd.DataFrame],
    error: ErrorType,
    base_path: str,
) -> None:
    """Train dataset to obtain the best hyperparameter set"""
    logger.debug("Training models to fit hyperparameters...")
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
    # Clean data to free memory
    del train_features
    del train_output
    del datasets
    # Get test datasets
    test_features, test_output = loading.datasets.load_test_datasets(
        base_path, config.NUM_FEATURES, config.NUM_PREDICTIONS
    )
    # Fit hyperparams
    for model in models_dict.values():
        model.fit(test_features, test_output)
    # Generate hyperparams_list to use in the friedman test
    hyperparams_list = [
        h for x in models_dict.values() for h in x.hyper_params.keys()
    ]
    logger.debug("Hyperparam list: %r", hyperparams_list)
    # Execute friedman test
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
        filename = os.path.join(base_path, "errors", "sum", f"{model}.csv.gz")
        sum_errors.append(
            pdu.read_csv(filename).set_index(["dataset", "timeseries"])
        )
    friedman_df = pd.concat(sum_errors, axis=1)
    columns = friedman_df.columns
    # Calculate friedman chi-square
    logger.debug("Friedman\n%r", friedman_df)
    f_test = friedmanchisquare(*[friedman_df[k] for k in columns])
    logger.debug("Friedmanchisquare result: %r", f_test)
    result_filename = os.path.join(base_path, "post-hoc", "result.txt")
    with open(result_filename, "w", encoding="utf8") as f_result:
        f_result.write(f"Friedmanchisquare result: {f_test}\n")
    # Calculate post-hoc if p_value < 0.05
    if f_test.pvalue < 0.05:
        new_columns = _generate_columns_dict(columns)
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
        legend_filename = os.path.join(base_path, "post-hoc", "legend.csv")
        with open(legend_filename, "w", encoding="utf8") as f_legend:
            f_legend.write(json.dumps(new_columns, indent=4))


def _generate_columns_dict(columns):
    columns_dict = {}
    alphabet = string.ascii_uppercase  # Alphabet

    for index, value in enumerate(columns):
        key = ""
        while index >= len(alphabet):  # Check if we need more than one letter
            index -= len(alphabet)
            key = alphabet[index] + key
        key = alphabet[index] + key
        columns_dict[key] = value

    return columns_dict
