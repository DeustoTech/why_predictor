"""Dataset loading functions"""
import copy
import glob
import logging
import math
import os
import random
import shutil
from multiprocessing import Pool
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd  # type: ignore

from .. import panda_utils as pdu

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
        series[name] = glob.glob(f"{basepath}/{name}/{dirname}/*.csv.gz")
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


def _generate_rolling_windows(
    data: pd.DataFrame, dataset_name: str, timeseries_name: str, window: int
) -> pd.DataFrame:
    matrix = []
    for i in range(len(data) - window):
        matrix.append(
            [dataset_name, timeseries_name, *list(data["kWh"][i : i + window])]
        )

    return pdu.DataFrame(matrix)


def _load_csv(
    counter: Tuple[int, int],
    names: Tuple[str, str],
    window: Tuple[int, int],
    train_test_ratio: float,
    dataset_path: Optional[str] = None,
) -> Tuple[str, str]:
    """Load and process timeseries CSV File"""
    dataset_name, filename = names
    timeseries = os.path.split(filename)[-1]
    logger.debug(
        "(%d/%d) loading: %s",
        counter[0],
        counter[1],
        filename,
    )
    data = pdu.read_csv(filename, usecols=["timestamp", "kWh"])
    # Set column as 'timestamp' (Pandas get it as str)
    data["timestamp"] = pd.to_datetime(data["timestamp"])
    # Use watios instead of kwh
    data["kWh"] = data["kWh"] * 1000
    # Filter out values, we only want one value per hour
    data = (
        data.set_index("timestamp")
        .resample("60T")
        .sum()
        .reset_index()
        .reindex(columns=data.columns)
    )
    # Sanity check
    if data["kWh"].max() > np.iinfo(np.uint16).max:
        return "", ""
    # convert 'kWh' to uint16
    data["kWh"] = data.kWh.apply(np.uint16)
    # Timeseries name
    timeseries_name = os.path.splitext(timeseries)[0]
    if timeseries_name.endswith("csv"):
        timeseries_name = os.path.splitext(timeseries_name)[0]
    # Generate rolling window values
    dtf = _generate_rolling_windows(
        data, dataset_name, timeseries_name, sum(window)
    )
    if dataset_path:
        dtf.to_csv(
            os.path.join(dataset_path, f"{os.path.split(filename)[1]}.gz"),
            index=False,
            compression={"method": "gzip", "compresslevel": 1, "mtime": 1},
        )
    # Split dataframes
    limit = math.ceil(len(dtf) * train_test_ratio)
    test = dtf.iloc[limit:]
    test.columns = [
        "dataset",
        "timeseries",
        *[f"col{i}" for i in range(1, sum(window) + 1)],
    ]
    dtf = dtf.iloc[:limit]  # train
    base_path = f"model-training/test/{dataset_name}/"
    test.iloc[:, : window[0] + 2].to_csv(
        os.path.join(base_path, "features", f"{timeseries}"),
        index=False,
        compression={"method": "gzip", "compresslevel": 1, "mtime": 1},
    )
    test.drop(test.iloc[:, 2 : window[0] + 2], axis=1).to_csv(
        os.path.join(base_path, "output", f"{timeseries}"),
        index=False,
        compression={"method": "gzip", "compresslevel": 1, "mtime": 1},
    )
    base_path = f"model-training/train/{dataset_name}/"
    dtf.iloc[:, : window[0] + 2].to_csv(
        os.path.join(base_path, "features", f"{timeseries}"),
        index=False,
        compression={"method": "gzip", "compresslevel": 1, "mtime": 1},
    )
    dtf.drop(dtf.iloc[:, 2 : window[0] + 2], axis=1).to_csv(
        os.path.join(base_path, "output", f"{timeseries}"),
        index=False,
        compression={"method": "gzip", "compresslevel": 1, "mtime": 1},
    )
    return dataset_name, timeseries


def _remove_files() -> None:
    base_path = "model-training"
    for test_train in ["test", "train"]:
        for feat_output in ["features", "output"]:
            for header_values in ["header", "values"]:
                rmpath = os.path.join(
                    base_path,
                    f"{test_train}_{feat_output}_{header_values}.csv.gz",
                )
                try:
                    os.remove(os.path.join(rmpath))
                except OSError:
                    pass


def load_files(
    training_set: Dict[str, List[str]],
    num_features: int,
    num_predictions: int,
    train_test_ratio: float,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load training set"""
    logger.info("Loading CSV files...")
    total_window = num_features + num_predictions
    _remove_files()
    for idxset, name in enumerate(training_set):
        logger.debug(
            "Dataset (%d/%d): %s", idxset + 1, len(training_set), name
        )
        for folder in ["train", "test"]:
            base_path = f"model-training/{folder}/{name}"
            if not os.path.exists(base_path):
                os.makedirs(os.path.join(base_path, "features"))
                os.makedirs(os.path.join(base_path, "output"))
        file_list = [
            (
                (i + 1, len(training_set[name])),
                (name, v),
                (num_features, num_predictions),
                train_test_ratio,
            )
            for i, v in enumerate(training_set[name])
        ]
        with Pool() as pool:
            _concat_csvs(pool.starmap(_load_csv, file_list))
            shutil.rmtree(os.path.join("model-training", "train", name))
    train_features = pdu.read_csv(
        "model-training/train_features.csv.gz",
        header=None,
        dtype={
            "0": "str",
            "1": "str",
            **{str(i): "uint16" for i in range(2, num_features + 2)},
        },
    )
    train_output = pdu.read_csv(
        "model-training/train_output_header.csv.gz",
        header=None,
        dtype={
            "0": "str",
            "1": "str",
            **{str(i): "uint16" for i in range(2, num_predictions + 2)},
        },
    )
    train_features.columns = [
        "dataset",
        "timeseries",
        *[f"col{i}" for i in range(1, num_features + 1)],
    ]
    train_output.columns = [
        "dataset",
        "timeseries",
        *[f"col{i}" for i in range(num_features + 1, total_window + 1)],
    ]
    logger.info("CSV files loaded.")
    return (train_features, train_output)


def _concat_csvs(dt_list: List[Tuple[str, str]]) -> None:
    for folder in ["train", "test"]:
        base_path = f"model-training/{folder}"
        for dataset, timeseries in dt_list:
            # Sanity check
            if dataset == "" or timeseries == "":
                continue
            # Process subfolder
            for subfolder in ["features", "output"]:
                dtf = pdu.read_csv(
                    os.path.join(
                        base_path, dataset, subfolder, f"{timeseries}"
                    )
                )
                dtf.to_csv(
                    os.path.join(
                        "model-training", f"{folder}_{subfolder}.csv.gz"
                    ),
                    mode="a",
                    header=False,
                    index=False,
                    compression={
                        "method": "gzip",
                        "compresslevel": 1,
                        "mtime": 1,
                    },
                )


def _get_train_test_dataframes(
    mydict: Dict[str, pd.DataFrame], train_ratio: float
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_dict = {}
    test_dict = {}
    for timeseries in mydict:
        dataset = mydict[timeseries]
        limit = math.ceil(len(dataset) * train_ratio)
        logger.debug("Limit %s: %d [%d]", timeseries, limit, len(dataset))
        train_dict[timeseries] = dataset.iloc[:limit]
        test_dict[timeseries] = dataset.iloc[limit:]
        # .reset_index(drop=True)
    train = pdu.concat(train_dict.values(), ignore_index=True)
    test = pdu.concat(test_dict.values(), ignore_index=True)
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
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split the FFORMA dataset in train and test"""
    # Create a FFORMA Dataset with dataset column
    limit = math.ceil(len(data) * train_ratio)
    test = data.iloc[limit:]
    base_path = "fforma-training/test/fforma/"
    if not os.path.exists(base_path):
        os.makedirs(os.path.join(base_path, "features"))
        os.makedirs(os.path.join(base_path, "output"))
    test.iloc[:, : num_features + 2].to_csv(
        os.path.join(base_path, "features", "dataset"),
        index=False,
        compression={"method": "gzip", "compresslevel": 1, "mtime": 1},
    )
    test.drop(test.iloc[:, 2 : num_features + 2], axis=1).to_csv(
        os.path.join(base_path, "output", "dataset"),
        index=False,
        compression={"method": "gzip", "compresslevel": 1, "mtime": 1},
    )
    data = data.iloc[:limit]  # train
    return (
        data.iloc[:, : num_features + 2],
        data.drop(data.iloc[:, 2 : num_features + 2], axis=1),
    )


def get_train_datasets(
    series: Dict[str, List[str]],
    percentage: float,
    num_features: int,
    num_predictions: int,
    train_test_ratio: float,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Generate train and test datasets"""
    # Select training set
    training_set, _ = select_training_set(series, percentage)
    # Load training set
    return load_files(
        training_set, num_features, num_predictions, train_test_ratio
    )


def process_and_save(
    training_set: Dict[str, List[str]],
    num_features: int,
    num_predictions: int,
) -> None:
    """Load CSV files, process and save them to files"""
    base_path = f"datasets/{num_features}x{num_predictions}/"
    if not os.path.exists(base_path):
        os.makedirs(base_path)
    for idxset, name in enumerate(training_set):
        dataset_path = os.path.join(base_path, name)
        if not os.path.exists(dataset_path):
            os.makedirs(dataset_path)
        logger.debug(
            "Dataset (%d/%d): %s", idxset + 1, len(training_set), name
        )
        file_list = [
            (
                (i + 1, len(training_set[name])),
                (name, v),
                (num_features, num_predictions),
                1.0,
                dataset_path,
            )
            for i, v in enumerate(training_set[name])
        ]
        with Pool() as pool:
            pool.starmap(_load_csv, file_list)
