"""Parsing utils for loading functions"""
import logging
import math
import os
import shutil
from multiprocessing import Pool
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd  # type: ignore

from ... import config
from ... import panda_utils as pdu

logger = logging.getLogger("logger")
POWER_EXC = "max kWh exceeded"


def generate_rolling_windows(
    data: pd.DataFrame, dataset_name: str, timeseries_name: str, window: int
) -> pd.DataFrame:
    """Convert dataset into a rolling window dataset: each original row into a
    column (window is the total number of windows)"""
    matrix = []
    for i in range(len(data) - window):
        matrix.append(
            [dataset_name, timeseries_name, *list(data["kWh"][i : i + window])]
        )

    dtf = pdu.DataFrame(matrix)
    dtf.columns = [
        "dataset",
        "timeseries",
        *[f"col{i}" for i in range(1, window + 1)],
    ]
    return dtf


def save_rolling_window_dataframe(dtf: pd.DataFrame, cache_name: str) -> None:
    """Save rolling window dataframe if required"""
    if config.SAVE_DATASETS:
        dtf.to_csv(
            cache_name,
            index=False,
            compression={"method": "gzip", "compresslevel": 1, "mtime": 1},
        )


def get_dataset_timeseries_name(
    names: Tuple[str, str]
) -> Tuple[str, str, str]:
    """Process names to obtain the filename, and the name of the dataset and
    the timeseries"""
    dataset_name, filename = names
    timeseries = os.path.split(filename)[-1]
    timeseries_name = os.path.splitext(timeseries)[0]
    if timeseries_name.endswith("csv"):
        timeseries_name = os.path.splitext(timeseries_name)[0]
    return filename, dataset_name, timeseries_name


def split_rolling_window_dtf_in_train_and_test(
    dtf: pd.DataFrame, train_test_ratio: float
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split the rolling window dataframe in train and test"""
    limit = math.ceil(len(dtf) * train_test_ratio)
    test = dtf.iloc[limit:]
    train = dtf.iloc[:limit]
    return train, test


def process_dataset_into_features_and_output(
    dtf: pd.DataFrame, base_path: str, timeseries: str, window: Tuple[int, int]
) -> None:
    """Process the train or test dataset saving it into features and output
    CSVs"""
    # Features
    logger.debug("Saving %s.csv.gz to %s%s", timeseries, base_path, "features")
    dtf.iloc[:, : window[0] + 2].to_csv(
        os.path.join(base_path, "features", f"{timeseries}.csv.gz"),
        index=False,
        compression={"method": "gzip", "compresslevel": 1, "mtime": 1},
    )
    # Output
    logger.debug("Saving %s to %s%s", timeseries, base_path, "output")
    dtf.drop(dtf.iloc[:, 2 : window[0] + 2], axis=1).to_csv(
        os.path.join(base_path, "output", f"{timeseries}.csv.gz"),
        index=False,
        compression={"method": "gzip", "compresslevel": 1, "mtime": 1},
    )


def process_raw_dataframe(data: pd.DataFrame) -> pd.DataFrame:
    """Process raw dataframe so it has one sample per hour and the power is in
    watts per hours instead of kWh"""
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
        raise ValueError("Max kWh in dataset")
    # convert 'kWh' to uint16
    data["kWh"] = data.kWh.apply("uint16")
    return data


def load_csv(
    counter: Tuple[int, int],
    names: Tuple[str, str],
    window: Tuple[int, int],
    train_test_ratio: float,
    root_path: str,
) -> Tuple[str, str]:
    """Load and process timeseries CSV File"""
    # Get filename and the name of both dataset and timeseries
    filename, dataset, timeseries = get_dataset_timeseries_name(names)
    logger.debug("(%d/%d) loading: %s", counter[0], counter[1], filename)
    # Load the dataframe
    cache_name = _generate_cache_name(dataset, timeseries, *window)
    if os.path.exists(cache_name):
        # Load from cache
        dtf = pdu.read_csv(cache_name)
    else:
        # Process raw file
        try:
            dtf = process_raw_dataframe(
                pdu.read_csv(filename, usecols=["timestamp", "kWh"])
            )
        except ValueError:
            return POWER_EXC, POWER_EXC
        # Generate rolling window values
        dtf = generate_rolling_windows(dtf, dataset, timeseries, sum(window))
        save_rolling_window_dataframe(dtf, cache_name)
    # Split dataframes
    train, test = split_rolling_window_dtf_in_train_and_test(
        dtf, train_test_ratio
    )
    # Process test dataframe
    process_dataset_into_features_and_output(
        test, f"{root_path}/datasets/test/{dataset}/", timeseries, window
    )
    # Process train dataframe
    process_dataset_into_features_and_output(
        train, f"{root_path}/datasets/train/{dataset}/", timeseries, window
    )
    return dataset, timeseries


def concat_csvs_in_file(
    dt_list: List[Tuple[str, str]], root_path: str
) -> None:
    """Concat CSV files"""
    for folder in ["train", "test"]:
        base_path = f"{root_path}/datasets/{folder}"
        # Generate datasets
        logger.debug("Generating %s dataset...", folder)
        for dataset, timeseries in dt_list:
            # Sanity check
            if POWER_EXC in (dataset, timeseries):
                logger.warning(
                    "Dataset is %s and timeseries is %s", dataset, timeseries
                )
                continue
            # Process subfolder
            for subfolder in ["features", "output"]:
                dtf = pdu.read_csv(
                    os.path.join(
                        base_path, dataset, subfolder, f"{timeseries}.csv.gz"
                    )
                )
                dtf.to_csv(
                    os.path.join(root_path, f"{folder}_{subfolder}.csv.gz"),
                    mode="a",
                    header=False,
                    index=False,
                    compression={
                        "method": "gzip",
                        "compresslevel": 1,
                        "mtime": 1,
                    },
                )


def generate_datasets(
    training_set: Dict[str, List[str]],
    window: Tuple[int, int],
    train_test_ratio: float,
    root_path: str,
) -> None:
    """Load training set"""
    logger.info("Loading CSV files...")
    for idxset, name in enumerate(training_set):
        logger.debug(
            "Dataset (%d/%d): %s", idxset + 1, len(training_set), name
        )
        _generate_dataset_train_test_folders(root_path, name)
        file_list = _generate_files_to_load(
            name, window, train_test_ratio, training_set[name], root_path
        )
        with Pool(_num_processes()) as pool:
            concat_csvs_in_file(pool.starmap(load_csv, file_list), root_path)
        if config.REMOVE_TRAIN_SETS:
            shutil.rmtree(os.path.join(root_path, "datasets", "train", name))


def _num_processes() -> Optional[int]:
    njobs = config.NJOBS
    cpu_count = os.cpu_count()
    num_cpus = cpu_count if cpu_count is not None else 1
    if njobs == -1:
        return None
    if njobs < -1:
        return max(num_cpus - njobs + 1, 1)
    return njobs


def _generate_dataset_train_test_folders(root_path: str, dataset: str) -> None:
    for folder in ["train", "test"]:
        base_path = f"{root_path}/datasets/{folder}/{dataset}"
        os.makedirs(os.path.join(base_path, "features"))
        os.makedirs(os.path.join(base_path, "output"))


def _generate_files_to_load(
    dataset: str,
    window: Tuple[int, int],
    ratio: float,
    files: List[str],
    root_path: str,
) -> List[
    Tuple[Tuple[int, int], Tuple[str, str], Tuple[int, int], float, str]
]:
    return [
        ((i + 1, len(files)), (dataset, v), window, ratio, root_path)
        for i, v in enumerate(files)
    ]


def _generate_cache_name(
    dataset: str,
    timeseries: str,
    num_features: int,
    num_predictions: int,
) -> str:
    # Generate the folder name
    base_path = os.path.join(
        config.DATASET_CACHE, f"{num_features}x{num_predictions}", dataset
    )
    # Generate the folder if it does not exist
    os.makedirs(base_path, exist_ok=True)
    # Return the name
    return os.path.join(base_path, f"{timeseries}.csv.gz")
