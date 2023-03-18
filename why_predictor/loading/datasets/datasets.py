"""Dataset loading functions"""
import glob
import logging
import os
import shutil
from typing import Dict, List, Tuple

import pandas as pd  # type: ignore

from ... import config
from ... import panda_utils as pdu
from .. import raw
from . import parser

logger = logging.getLogger("logger")


def load_train_datasets(
    base_path: str, num_features: int, num_predictions: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load train datasets (features & output)"""
    logger.debug("Loading train datasets...")
    return load_datasets(base_path, "train", num_features, num_predictions)


def load_test_datasets(
    base_path: str, num_features: int, num_predictions: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load train datasets (features & output)"""
    logger.debug("Loading test datasets...")
    return load_datasets(base_path, "test", num_features, num_predictions)


def get_train_datasets(
    series: Dict[str, List[str]],
    percentage: float,
    window: Tuple[int, int],
    train_test_ratio: float,
    root_path: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Generate train and test datasets"""
    num_features, num_predictions = window
    # Select training set
    training_set, _ = raw.select_training_set(series, percentage)
    # Generate datasets
    logger.debug("Generating datasets...")
    parser.generate_datasets(
        training_set,
        (num_features, num_predictions),
        train_test_ratio,
        root_path,
    )
    # Load training set
    return load_train_datasets(root_path, num_features, num_predictions)


def get_train_datasets_from_initial_path(
    series: Dict[str, List[str]],
    percentage: float,
    window: Tuple[int, int],
    initial_path: str,
    root_path: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Generate train and test datasets (train dataset from initial path)"""
    num_features, num_predictions = window
    # Generate train datasets
    logger.debug("Generating test dataset...")
    training_set = {}
    initial_name = os.path.split(initial_path)[-1]
    training_set[initial_name] = glob.glob(
        os.path.join(initial_path, "*.csv.gz")
    )
    parser.generate_datasets(
        training_set,
        (num_features, num_predictions),
        1,
        root_path,
    )
    # Select training set
    test_set, _ = raw.select_training_set_filtered(
        series, percentage, training_set
    )
    logger.debug("Generating test dataset...")
    parser.generate_datasets(
        test_set,
        (num_features, num_predictions),
        0,
        root_path,
    )
    # Return train datasets
    return load_train_datasets(root_path, num_features, num_predictions)


def process_and_save(
    training_set: Dict[str, List[str]],
    num_features: int,
    num_predictions: int,
) -> None:
    """Load CSV files, process and save them to files"""
    old_value = config.SAVE_DATASETS
    config.SAVE_DATASETS = True
    parser.generate_datasets(
        training_set, (num_features, num_predictions), 1, "/tmp"
    )
    config.SAVE_DATASETS = old_value
    shutil.rmtree("/tmp/datasets")


def load_datasets(
    base_path: str,
    train_test_type: str,
    num_features: int,
    num_predictions: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load datasets"""
    total_window = num_features + num_predictions
    # Features
    logger.debug("Reading %s_features file...", train_test_type)
    feat_types = {
        "dataset": "str",
        "timeseries": "str",
        **{f"col{i}": "uint16" for i in range(1, num_predictions + 1)},
    }
    set_features = pdu.read_csv(
        f"{base_path}/{train_test_type}_features.csv.gz",
        # header=None,
        dtype=feat_types,
    )
    # Output
    logger.debug("Reading %s_output file...", train_test_type)
    out_types = {
        "dataset": "str",
        "timeseries": "str",
        **{f"col{i}": "uint16" for i in range(1, total_window + 1)},
    }
    set_output = pdu.read_csv(
        f"{base_path}/{train_test_type}_output.csv.gz",
        # header=None,
        dtype=out_types,
    )
    logger.info("CSV files loaded.")
    return (set_features, set_output)
