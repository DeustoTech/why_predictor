"""Unit tests for loading_functions module"""
import os
import shutil
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd  # type: ignore

from why_predictor import loading


class LoadingFunctionsTest(unittest.TestCase):
    """loading functions test"""

    def setUp(self) -> None:
        # Init
        root_path = "tests/results"
        shutil.rmtree(root_path, ignore_errors=True)
        for dirname in ["features", "output"]:
            base_path = os.path.join(root_path, "datasets")
            for folder in ["train", "test"]:
                for dataset in ["mydataset", "dataset2"]:
                    folder_path = os.path.join(
                        base_path, folder, dataset, dirname
                    )
                    os.makedirs(folder_path)
        os.makedirs(os.path.join(root_path, "cache/72x6/test1"))
        self.root_path = root_path

    def tearDown(self) -> None:
        shutil.rmtree(self.root_path)

    @patch("why_predictor.config.DATASET_CACHE", "tests/results/cache")
    def test_process_and_save(self) -> None:
        """test process and save"""
        # Init
        cache_name = os.path.join(
            self.root_path, "cache/72x6/test1/003.csv.gz"
        )
        training_set = {"test1": ["tests/data/test1/imp_csv/003.csv.gz"]}
        # Execute
        loading.datasets.process_and_save(training_set, 72, 6)
        # Validate
        self.assertTrue(os.path.exists(cache_name))

    def test_load_datasets(self) -> None:
        """test load_datasets"""
        # Init
        num_features = 72
        num_predictions = 6
        os.makedirs("tests/results", exist_ok=True)
        dtf = pd.read_csv("tests/data/rolling.csv.gz")
        for folder in ["train", "test"]:
            dtf.iloc[:, : num_features + 2].to_csv(
                f"tests/results/{folder}_features.csv.gz"
            )
            dtf.drop(dtf.iloc[:, 2 : num_features + 2], axis=1).to_csv(
                f"tests/results/{folder}_output.csv.gz"
            )
        # Execute
        feat, out = loading.datasets.load_datasets(
            "tests/results", "train", num_features, num_predictions
        )
        for col in range(1, num_features + 1):
            self.assertEqual(feat[f"col{col}"].dtype, np.uint16)
        for col in range(num_features + 1, num_predictions + num_features + 1):
            self.assertEqual(out[f"col{col}"].dtype, np.uint16)
