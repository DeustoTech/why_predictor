""" WHY predictor
"""
import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from sklearn.linear_model import LinearRegression  # type: ignore

from .load_sets import (
    find_csv_files,
    load_files,
    select_training_set,
    split_dataset_in_train_and_test,
)

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)7s:  %(message)s",
)
logger = logging.getLogger("logger")
load_dotenv(dotenv_path=Path("./config.env"))


def generate_parser() -> argparse.ArgumentParser:
    """Generate parser for command line arguments"""
    parser = argparse.ArgumentParser(
        prog="python -m why-predictor",
        description="WHY Predictor",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument(
        "--base-path-dataset",
        dest="dataset_basepath",
        default=os.getenv("DATASET_BASEPATH"),
    )
    parser.add_argument(
        "--dataset-dir-name",
        dest="dataset_dir_name",
        default=os.getenv("DATASET_DIRNAME"),
    )
    parser.add_argument(
        "--percentage-training",
        dest="training_percentage",
        type=float,
        default=os.getenv("TRAINING_PERCENTAGE"),
    )
    return parser


def execute(args: argparse.Namespace) -> None:
    """Execute"""
    basepath = args.dataset_basepath
    dirname = args.dataset_dir_name
    series = find_csv_files(basepath, dirname)
    # Select training set
    training_set, _ = select_training_set(series, args.training_percentage)
    # Load training set
    data = load_files(training_set)
    # Train and test datasets
    train, test = split_dataset_in_train_and_test(data)
    train_features = train.iloc[:, :72]
    train_output = train.iloc[:, 72:]
    test_features = test.iloc[:, :72]
    test_output = test.iloc[:, 72:]
    model = LinearRegression()
    linear_model = model.fit(train_features, train_output)
    predictions = linear_model.predict(test_features)
    logger.debug("Accuracy: %r", model.score(test_features, test_output))
    logger.info(
        "MAPE linear regression:\n%r",
        np.absolute((test_output - predictions) / test_output).sum()
        / len(test_output),
    )


def main() -> None:
    """Main"""
    parser = generate_parser()

    # Parse args
    args = parser.parse_args()

    # Set logger
    if args.verbose:
        logger.setLevel(logging.DEBUG)

    print(args)
    logger.debug("Args: %r", args)
    execute(args)


if __name__ == "__main__":
    main()
