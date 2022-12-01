"""Dataset loading functions"""
import copy
import glob
import logging
import os
import random
from typing import Dict, List, Tuple

import pandas as pd  # type: ignore

logger = logging.getLogger("logger")


def get_datasets(basepath: str, dirname: str) -> List[str]:
    """Get the list of valid datasets"""
    valid_datasets = []
    for file in glob.glob(f"{basepath}/*/{dirname}"):
        valid_datasets.append(os.path.split(file)[0].split("/")[-1])
    logger.info(
        "There are %d valid datasets: %r", len(valid_datasets), valid_datasets
    )
    return valid_datasets


def find_csv_files(basepath: str, dirname: str) -> Dict[str, List[str]]:
    """Find CSV files in datasets"""
    logger.info("Finding CSV files...")
    series: Dict[str, List[str]] = {}
    datasets = get_datasets(basepath, dirname)
    for name in datasets:
        series[name] = glob.glob(f"{basepath}/{name}/{dirname}/*.csv")
    logger.info("CSV files found.")
    return series


def select_training_set(
    series: Dict[str, List[str]], percentage: float
) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    """Select Training and No-training sets"""
    logger.info("Selecting files for training set...")
    no_training_set = copy.deepcopy(series)
    training_set: Dict[str, List[str]] = {key: [] for key in series}
    for name in no_training_set:
        num = percentage * len(no_training_set[name])
        while len(training_set[name]) < num:
            training_set[name].append(
                no_training_set[name].pop(
                    random.randint(0, len(no_training_set[name]) - 1)
                )
            )
    logger.info("Files for training set selected.")
    return training_set, no_training_set


def load_files(training_set: Dict[str, List[str]]) -> pd.DataFrame:
    """Load training set"""
    logger.info("Loading CSV files...")
    matrix = []
    for idxset, name in enumerate(training_set):
        logger.debug(
            "Dataset (%d/%d): %s", idxset + 1, len(training_set), name
        )
        for idx, filename in enumerate(training_set[name]):
            logger.debug(
                "(%d/%d) loading: %s",
                idx + 1,
                len(training_set[name]),
                filename,
            )
            data = pd.read_csv(filename)
            # Set column as 'timestamp' (Pandas get it as str)
            data["timestamp"] = pd.to_datetime(data["timestamp"])
            # Filter out values, we only want one value per hour
            data = (
                data.set_index("timestamp")
                .resample("60T")
                .sum()
                .reset_index()
                .reindex(columns=data.columns)
            )
            # Generate rolling window values
            for i in range(len(data) - 78):
                matrix.append(list(data["kWh"][i : i + 78]))
    logger.debug("Generating DataFrame...")
    data = pd.DataFrame(matrix, columns=[f"col{i}" for i in range(1, 79)])
    logger.info("CSV files loaded.")
    return data


def split_dataset_in_train_and_test(
    data: pd.DataFrame, train_ratio: float
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split the dataset in train and test"""
    logger.info("Generating train and test sets...")
    limit = int(len(data) * train_ratio)
    train = data.iloc[:limit]
    test = data.iloc[limit:]
    logger.info("Train\n%r", train.head())
    logger.info("Train shape: %r", train.shape)
    logger.info("Test\n%r", test.head())
    logger.info("Test shape: %r", test.shape)
    logger.info(test.shape)
    train_features = train.iloc[:, :72]
    train_output = train.iloc[:, 72:]
    test_features = test.iloc[:, :72]
    test_output = test.iloc[:, 72:]
    return (train_features, train_output, test_features, test_output)
