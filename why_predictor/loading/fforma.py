"""Loading FFORMA module"""
import logging
import math
import os
from typing import Dict, Tuple

import numpy as np
import pandas as pd  # type: ignore

from .. import config
from .. import panda_utils as pdu
from ..errors import ErrorType
from ..models import BasicModel

logger = logging.getLogger("logger")


def generate_fforma_datasets(
    models_dict: Dict[str, BasicModel],
    datasets: Tuple[pd.DataFrame, pd.DataFrame],
    train_test_ratio: float,
    base_path: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Generate FFORMA datasets returning train datasets"""
    logger.debug("Generating FFORMA datasets...")
    features, output = datasets
    return split_fforma_in_train_test(
        generate_fforma_features(features),
        generate_fforma_output(models_dict, features, output),
        train_test_ratio,
        base_path,
    )


def generate_fforma_features(feat: pd.DataFrame) -> pd.DataFrame:
    """load FFORMA context"""
    full_context = load_context()
    logger.debug("Loading fforma features...")
    return full_context[
        full_context.index.isin(
            [
                (x.dataset, x.timeseries)
                for x in feat.iloc[:, :2].drop_duplicates().itertuples()
            ]
        )
    ].reset_index()


def load_context(base_path: str = ".") -> pd.DataFrame:
    """load full context dataframe"""
    logger.debug("Loading full context...")
    context = pdu.read_csv(os.path.join(base_path, "context.csv"))
    context.timeseries = context.timeseries.astype("string")
    context.set_index(["dataset", "timeseries"], inplace=True)
    logger.debug("Context loaded.")
    return context


def generate_fforma_output(
    models_dict: Dict[str, BasicModel],
    features: pd.DataFrame,
    output: pd.DataFrame,
) -> pd.DataFrame:
    """Generate FFORMA output"""
    logger.debug("Generating fforma output...")
    dtfs = []
    median_value = np.nanmean(features.iloc[:, 2:])
    for model in models_dict.values():
        dtfs.append(
            model.calculate_timeseries_errors_dataframe(
                (features, output),
                ErrorType[config.ERROR_TYPE_MODELS],
                median_value,
            )
        )
    logger.debug("FFORMA output generated, concatenating in one DF")
    return pdu.concat(dtfs, axis=1).reset_index()


def clean_bad_models(fforma_out: pd.DataFrame) -> None:
    """Clean models where their output is NaN or infinite"""
    logger.debug("Cleaning FFORMA bad models...")
    for col in fforma_out.columns[2:]:
        if (
            fforma_out[col].isin([math.inf]).values.any()
            or fforma_out[col].isna().values.any()
        ):
            fforma_out.drop(col, axis=1, inplace=True)
    logger.debug("FFORMA bad models cleaned.")


def split_fforma_in_train_test(
    fforma_feat: pd.DataFrame,
    fforma_out: pd.DataFrame,
    train_test_ratio: float,
    base_path: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split the FFORMA datasets in train and test"""
    logger.debug("Splitting FFORMA in train & test...")
    # OUTPUT
    list_train = []
    list_test = []
    for _, dtf in fforma_out.groupby(["dataset"]):
        value = math.ceil(dtf.shape[0] * train_test_ratio)
        list_train.append(dtf.iloc[:value])
        list_test.append(dtf.iloc[value:])
    # - generate and save test output dataset
    logger.debug("Saving FFORMA test-output dataset...")
    pdu.concat(list_test).to_csv(
        os.path.join(base_path, "test_output.csv.gz"), index=False
    )
    # - generate and save train output dataset
    logger.debug("Saving FFORMA train-output dataset...")
    fforma_train_out = pdu.concat(list_train)
    fforma_train_out.to_csv(
        os.path.join(base_path, "train_output.csv.gz"), index=False
    )
    # FEATURES
    list_train = []
    list_test = []
    for _, dtf in fforma_feat.groupby(["dataset"]):
        value = math.ceil(dtf.shape[0] * train_test_ratio)
        list_train.append(dtf.iloc[:value])
        list_test.append(dtf.iloc[value:])
    # - generate and save test features dataset
    logger.debug("Saving FFORMA test-features dataset...")
    pdu.concat(list_test).to_csv(
        os.path.join(base_path, "test_features.csv.gz"), index=False
    )
    # - generate and save train features dataset
    logger.debug("Saving FFORMA train-features dataset...")
    fforma_train_feat = pdu.concat(list_train)
    fforma_train_feat.to_csv(
        os.path.join(base_path, "train_features.csv.gz"), index=False
    )
    # Return train datasets
    logger.debug("FFORMA splitted in train & test.")
    return fforma_train_feat, fforma_train_out


def load_fforma_train_datasets(
    base_path: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load FFORMA train datasets"""
    return load_fforma_datasets(base_path, "train")


def load_fforma_test_datasets(
    base_path: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load FFORMA test datasets"""
    return load_fforma_datasets(base_path, "test")


def load_fforma_datasets(
    base_path: str, ftype: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load FFORMA datasets (test or train, based on ftype)"""
    feat = pdu.read_csv(os.path.join(base_path, f"{ftype}_features.csv.gz"))
    output = pdu.read_csv(os.path.join(base_path, f"{ftype}_output.csv.gz"))
    return feat, output
