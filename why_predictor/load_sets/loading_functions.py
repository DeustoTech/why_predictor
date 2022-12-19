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


def load_files(
    training_set: Dict[str, List[str]],
    num_features: int,
    num_predictions: int,
) -> pd.DataFrame:
    """Load training set"""
    logger.info("Loading CSV files...")
    matrix = []
    total_window = num_features + num_predictions
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
            # Timeseries name
            timeseries_name = (
                f"{name}_"
                + f"{os.path.splitext(os.path.split(filename)[-1])[0]}"
            )
            # Generate rolling window values
            for i in range(len(data) - total_window):
                matrix.append(
                    [timeseries_name, *list(data["kWh"][i : i + total_window])]
                )
    logger.debug("Generating DataFrame...")
    data = pd.DataFrame(
        matrix,
        columns=[
            "timeseries",
            *[f"col{i}" for i in range(1, total_window + 1)],
        ],
    )
    logger.info("CSV files loaded.")
    return data


def _get_train_test_dataframes(
    mydict: Dict[str, pd.DataFrame], train_ratio: float
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_dict = {}
    test_dict = {}
    for timeseries in mydict:
        dataset = mydict[timeseries]
        limit = int(len(dataset) * train_ratio)
        logger.debug("Limit %s: %d", timeseries, limit)
        train_dict[timeseries] = dataset.iloc[:limit]
        test_dict[timeseries] = dataset.iloc[limit:]
        # .reset_index(drop=True)
    train = pd.concat(train_dict.values(), ignore_index=True)
    test = pd.concat(test_dict.values(), ignore_index=True)
    return (train, test)


def split_dataset_in_train_and_test(
    data: pd.DataFrame,
    train_ratio: float,
    num_features: int,
    num_head_columns: int = 1,
    groupby_name: str = "timeseries",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split the dataset in train and test"""
    logger.info("Generating train and test sets...")
    train, test = _get_train_test_dataframes(
        dict(tuple(data.groupby(groupby_name))), train_ratio
    )
    logger.info("Train\n%r", train.head())
    logger.info("Train shape: %r", train.shape)
    logger.info("Test\n%r", test.head())
    logger.info("Test shape: %r", test.shape)
    logger.info(test.shape)
    train_features = train.iloc[:, : num_features + num_head_columns]
    train_output = train.drop(
        train.iloc[:, num_head_columns:num_features], axis=1
    )
    test_features = test.iloc[:, : num_features + num_head_columns]
    test_output = test.drop(
        test.iloc[:, num_head_columns:num_features], axis=1
    )
    return (train_features, train_output, test_features, test_output)


def split_fforma_in_train_and_test(
    data: pd.DataFrame, train_ratio: float, num_features: int
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split the FFORMA dataset in train and test"""
    # Create a FFORMA Dataset with dataset column
    newdata = pd.DataFrame(
        [x[0] for x in data.iloc[:, 0].str.split("_").tolist()]
    )
    newdata = pd.concat([newdata, data], ignore_index=True, axis=1)
    newdata.columns = ["dataset", *data.columns]
    logger.debug("FFORMA with dataset column:\n%r", newdata)
    (
        train_features,
        train_output,
        test_features,
        test_output,
    ) = split_dataset_in_train_and_test(
        newdata, train_ratio, num_features, 2, "dataset"
    )
    return (
        train_features.drop(["dataset"], axis=1),
        train_output.drop(["dataset"], axis=1),
        test_features.drop(["dataset"], axis=1),
        test_output.drop(["dataset"], axis=1),
    )


def get_train_and_test_datasets(
    series: Dict[str, List[str]],
    percentage: float,
    num_features: int,
    num_predictions: int,
    train_test_ratio: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Generate train and test datasets"""
    # Select training set
    training_set, _ = select_training_set(series, percentage)
    # Load training set
    data = load_files(training_set, num_features, num_predictions)
    # Train and test datasets
    return split_dataset_in_train_and_test(
        data, train_test_ratio, num_features
    )
