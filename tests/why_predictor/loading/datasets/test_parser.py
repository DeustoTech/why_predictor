"""Test loading.datasets.parser functions"""
import logging
import os
import shutil
import unittest
from datetime import datetime
from unittest.mock import patch

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from parameterized import parameterized  # type: ignore

import why_predictor.panda_utils as pdu
from why_predictor import loading


def _read_csv(filename: str = "tests/data/test1/imp_csv/001.csv.gz"):
    return pd.read_csv(filename, usecols=["timestamp", "kWh"])


class DataFrameLoadingParserTest(unittest.TestCase):
    """loading.dataset.parser test"""

    def setUp(self):
        pass

    def test_get_dataset_timeseries_name(self):
        """get_dataset_timeseries_name test"""
        names = ["test1", "tests/data/test1/imp_csv/001.csv.gz"]
        (
            filename,
            dataset,
            timeseries,
        ) = loading.datasets.parser.get_dataset_timeseries_name(names)
        self.assertEqual(filename, names[1])
        self.assertEqual(dataset, "test1")
        self.assertEqual(timeseries, "001")

    def test_process_raw_dataframe(self):
        """test process_raw_dataframe"""
        result = loading.datasets.parser.process_raw_dataframe(_read_csv())
        self.assertEqual(result.iloc[0, 0], datetime(2008, 5, 14, 1))
        self.assertEqual(result.iloc[0, 1], 250)
        self.assertEqual(result.iloc[1, 0], datetime(2008, 5, 14, 2))
        self.assertEqual(result.iloc[1, 1], 200)
        self.assertEqual(result.timestamp.dtype, np.dtype("<M8[ns]"))
        self.assertEqual(result.kWh.dtype, np.uint16)

    def test_process_raw_dataframe_max_power(self):
        """test process_raw_dataframe with value with max power exceeded"""
        dtf = _read_csv("tests/data/maxpower.csv.gz")
        self.assertRaises(
            ValueError, loading.datasets.parser.process_raw_dataframe, dtf
        )

    @parameterized.expand([(78,), (66,)])
    def test_generate_rolling_window_values(self, window):
        """test generate_rolling_windows"""
        raw = loading.datasets.parser.process_raw_dataframe(_read_csv())
        dtf = loading.datasets.parser.generate_rolling_windows(
            raw, "mydataset", "mytimeseries", window
        )
        self.assertEqual(set(dtf.dataset), {"mydataset"})
        self.assertEqual(set(dtf.timeseries), {"mytimeseries"})
        self.assertEqual(dtf.shape[1], window + 2)
        self.assertEqual(dtf.shape[0], raw.shape[0] - window)
        value = dtf.iloc[0, window + 1]
        for col in range(2, window + 2):
            self.assertEqual(dtf[f"col{col-1}"].dtype, np.uint16)
            self.assertEqual(dtf.iloc[col - 2, window + 3 - col], value)

    @parameterized.expand(
        [
            (0, 0, 19717),
            (0.33, 6507, 13210),
            (0.5, 9859, 9858),
            (0.8, 15774, 3943),
            (1, 19717, 0),
        ]
    )
    def test_split_rolling_window_dtf_in_train_and_test(
        self, ratio, expected_train_len, expected_test_len
    ):
        """test split_rolling_window_dtf_in_train_and_test"""
        num = 78
        dtf = pdu.read_csv("tests/data/rolling.csv.gz")
        (
            train,
            test,
        ) = loading.datasets.parser.split_rolling_window_dtf_in_train_and_test(
            dtf, ratio
        )
        self.assertEqual(train.shape, (expected_train_len, num + 2))
        self.assertEqual(test.shape, (expected_test_len, num + 2))
        self.assertEqual(
            list(test.columns),
            ["dataset", "timeseries", *[f"col{i}" for i in range(1, num + 1)]],
        )
        for col in range(1, num + 1):
            self.assertEqual(train[f"col{col}"].dtype, np.uint16)
            self.assertEqual(test[f"col{col}"].dtype, np.uint16)


class DataFrameLoadingParserTestToDisk(unittest.TestCase):
    """load_set.loading.datasets.parser test"""

    def setUp(self):
        # Init
        self.dtf = pdu.read_csv("tests/data/rolling.csv.gz")
        shutil.rmtree("tests/results", ignore_errors=True)
        root_path = "tests/results"
        for dirname in ["features", "output"]:
            base_path = os.path.join(root_path, "datasets")
            for folder in ["train", "test"]:
                for dataset in ["mydataset", "dataset2"]:
                    folder_path = os.path.join(
                        base_path, folder, dataset, dirname
                    )
                    os.makedirs(folder_path)
        self.root_path = root_path

    def tearDown(self):
        # Clean
        shutil.rmtree(self.root_path)

    @parameterized.expand([([72, 6],), ([66, 12],)])
    def test_process_dataset_into_features_and_output(self, window):
        """test process_dataset_into_features_and_output"""
        base_path = os.path.join(self.root_path, "datasets/test/mydataset")
        # Execute function
        loading.datasets.parser.process_dataset_into_features_and_output(
            self.dtf, base_path, "timeseries", window
        )
        # Test features
        feat = pd.read_csv(
            os.path.join(base_path, "features", "timeseries.csv.gz")
        )
        self.assertEqual(feat.shape, (self.dtf.shape[0], window[0] + 2))
        # Test output
        out = pd.read_csv(
            os.path.join(base_path, "output", "timeseries.csv.gz")
        )
        self.assertEqual(out.shape, (self.dtf.shape[0], window[1] + 2))

    @patch("why_predictor.config.SAVE_DATASETS", True)
    @patch("why_predictor.config.DATASET_CACHE", "tests/results/cache")
    def test_save_rolling_window_dataframe_when_save_datasets_is_true(self):
        """test save_rolling_window_dataframe (SAVE_DATASETS=True)"""
        # Init
        num_features = 72
        num_predictions = 6
        base_path = os.path.join(
            "tests/results/cache",
            f"{num_features}x{num_predictions}",
            "test1",
        )
        os.makedirs(base_path, exist_ok=True)
        cache_name = os.path.join(base_path, "series1.csv.gz")
        # Execute function
        loading.datasets.parser.save_rolling_window_dataframe(
            self.dtf, cache_name
        )
        # Validate
        self.assertTrue(os.path.exists(cache_name))

    @patch("why_predictor.config.SAVE_DATASETS", False)
    @patch("why_predictor.config.DATASET_CACHE", "tests/results/cache")
    def test_save_rolling_window_dataframe_when_save_datasets_is_false(self):
        """test save_rolling_window_dataframe (SAVE_DATASETS=False)"""
        # Init
        num_features = 72
        num_predictions = 6
        base_path = os.path.join(
            "tests/results/cache",
            f"{num_features}x{num_predictions}",
            "test1",
        )
        os.makedirs(base_path, exist_ok=True)
        cache_name = os.path.join(base_path, "series1.csv.gz")
        # Execute function
        loading.datasets.parser.save_rolling_window_dataframe(
            self.dtf, cache_name
        )
        # Validate
        self.assertFalse(os.path.exists(cache_name))

    def test_concat_csvs_in_file(self):
        """test concat_csvs_in_file"""
        logging.getLogger("logger").disabled = True
        # Copy dataset-timeseries 1
        shutil.copy(
            "tests/data/rolling.csv.gz",
            "tests/results/datasets/train/mydataset/features/"
            + "mytimeseries.csv.gz",
        )
        shutil.copy(
            "tests/data/rolling.csv.gz",
            "tests/results/datasets/train/mydataset/output/mytimeseries.csv.gz",
        )
        shutil.copy(
            "tests/data/rolling.csv.gz",
            "tests/results/datasets/test/mydataset/features/"
            + "mytimeseries.csv.gz",
        )
        shutil.copy(
            "tests/data/rolling.csv.gz",
            "tests/results/datasets/test/mydataset/output/mytimeseries.csv.gz",
        )
        # Copy dataset-timeseries 2
        shutil.copy(
            "tests/data/rolling2.csv.gz",
            "tests/results/datasets/train/dataset2/features/timeseries1.csv.gz",
        )
        shutil.copy(
            "tests/data/rolling2.csv.gz",
            "tests/results/datasets/train/dataset2/output/timeseries1.csv.gz",
        )
        shutil.copy(
            "tests/data/rolling2.csv.gz",
            "tests/results/datasets/test/dataset2/features/timeseries1.csv.gz",
        )
        shutil.copy(
            "tests/data/rolling2.csv.gz",
            "tests/results/datasets/test/dataset2/output/timeseries1.csv.gz",
        )
        # Execute concat
        loading.datasets.parser.concat_csvs_in_file(
            [
                ("mydataset", "mytimeseries"),
                ("max kWh exceeded", "max kWh exceeded"),
                ("dataset2", "timeseries1"),
            ],
            self.root_path,
        )
        # Test
        for folder in ["train", "test"]:
            for subfolder in ["features", "output"]:
                filename = os.path.join(
                    self.root_path, f"{folder}_{subfolder}.csv.gz"
                )
                self.assertTrue(os.path.exists(filename))
                dtf = pd.read_csv(filename, header=None)
                self.assertEqual(dtf.shape, (20000, 80))
