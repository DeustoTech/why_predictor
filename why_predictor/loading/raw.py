"""Dataset loading functions - loading"""
import copy
import glob
import logging
import os
import random
from typing import Dict, List, Tuple

logger = logging.getLogger("logger")


def retrieve_csv_series_dict(
    basepath: str, dirname: str
) -> Dict[str, List[str]]:
    """Find CSV files in datasets and retrieve a dict with the timeseries per
    dataset"""
    logger.info("Finding CSV files...")
    series: Dict[str, List[str]] = {}
    datasets = get_datasets_names(basepath, dirname)
    for name in datasets:
        series[name] = glob.glob(f"{basepath}/{name}/{dirname}/*.csv.gz")
    logger.info("CSV files found.")
    return series


def get_datasets_names(basepath: str, dirname: str) -> List[str]:
    """Get the list of valid datasets"""
    valid_datasets = []
    for file in glob.glob(f"{basepath}/*/{dirname}"):
        valid_datasets.append(os.path.split(file)[0].split("/")[-1])
    logger.info(
        "There are %d valid datasets: %r", len(valid_datasets), valid_datasets
    )
    return valid_datasets


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


def select_training_set_filtered(
    series: Dict[str, List[str]],
    percentage: float,
    filtered_files: Dict[str, List[str]],
) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    """Select Training and No-training sets"""
    logger.info("Selecting files for training set...")
    no_training_set = copy.deepcopy(series)
    training_set: Dict[str, List[str]] = {key: [] for key in series}
    filtered = {
        k: [os.path.split(f)[-1] for f in filtered_files[k]]
        for k in filtered_files
    }
    for name in no_training_set:
        num = percentage * len(no_training_set[name])
        while len(training_set[name]) < num:
            candidate = no_training_set[name].pop(
                random.randint(0, len(no_training_set[name]) - 1)
            )
            if name not in filtered or candidate not in filtered[name]:
                training_set[name].append(candidate)
    logger.info("Files for training set selected.")
    return training_set, no_training_set
