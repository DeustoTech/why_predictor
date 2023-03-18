"""Tests for loading.fforma module"""
import os
import random
import shutil
import unittest
from unittest.mock import patch

import numpy as np  # type: ignore
import pandas as pd  # type: ignore

from why_predictor import loading
from why_predictor.models import Models

CONTEXT_COLUMNS = [
    "mean",
    "variance",
    "skewness",
    "kurtosis",
    "minimum",
    "maximum",
    "median",
    "quartile_1",
    "quartile_3",
    "decile_1",
    "decile_2",
    "decile_3",
    "decile_4",
    "decile_6",
    "decile_7",
    "decile_8",
    "decile_9",
    "iqr",
    "iqr_outlier_pc",
    "sum2",
    "rel_mean_00h04hspr_drm",
    "rel_mean_04h08hspr_drm",
    "rel_mean_08h12hspr_drm",
    "rel_mean_12h16hspr_drm",
    "rel_mean_16h20hspr_drm",
    "rel_mean_20h00hspr_drm",
    "rel_mean_00h04hsum_drm",
    "rel_mean_04h08hsum_drm",
    "rel_mean_08h12hsum_drm",
    "rel_mean_12h16hsum_drm",
    "rel_mean_16h20hsum_drm",
    "rel_mean_20h00hsum_drm",
    "rel_mean_00h04haut_drm",
    "rel_mean_04h08haut_drm",
    "rel_mean_08h12haut_drm",
    "rel_mean_12h16haut_drm",
    "rel_mean_16h20haut_drm",
    "rel_mean_20h00haut_drm",
    "rel_mean_00h04hwin_drm",
    "rel_mean_04h08hwin_drm",
    "rel_mean_08h12hwin_drm",
    "rel_mean_12h16hwin_drm",
    "rel_mean_16h20hwin_drm",
    "rel_mean_20h00hwin_drm",
    "rel_mean_weekday_pday",
    "ac_day_1",
    "ac_day_2",
    "ac_day_3",
]


class LoadingFFORMAFunctionsTest(unittest.TestCase):
    """loading.fforma functions test"""

    def setUp(self):
        os.makedirs("tests/results/errors/raw")
        os.makedirs("tests/results/errors/sum")
        os.makedirs("tests/results/predictions")

    def tearDown(self):
        shutil.rmtree("tests/results")

    def test_load_context(self):
        """test loading.fforma.load_context"""
        context = loading.fforma.load_context("tests/data")
        self.assertEqual(context.shape, (81, 48))
        self.assertEqual(context.columns.to_list(), CONTEXT_COLUMNS)

    @patch(
        "why_predictor.loading.fforma.load_context",
        return_value=loading.fforma.load_context("tests/data"),
    )
    def test_generate_fforma_features(self, _):
        """test loading.fforma.generate_fforma_features"""
        # Init
        dtf = pd.read_csv("tests/data/dataset.csv.gz")
        # Execute
        fforma_feat = loading.fforma.generate_fforma_features(dtf.iloc[:, :74])
        # Validate
        self.assertEqual(fforma_feat.shape, (10, 50))
        self.assertEqual(
            fforma_feat.columns.to_list(),
            ["dataset", "timeseries", *CONTEXT_COLUMNS],
        )

    def test_generate_fforma_output(self):
        """test loading.fforma.generate_fforma_output"""
        # Init
        shutil.copytree("tests/data/models", "tests/results/models")
        models_dict = {
            "SHIFT_LR": Models["SHIFT_LR"].value({}, "tests/results"),
            "SHIFT_SVR": Models["SHIFT_SVR"].value(
                {
                    "epsilon": 0.0,
                    "tol": 0.0001,
                    "C": 1.0,
                    "loss": "epsilon_insensitive",
                },
                "tests/results",
            ),
        }
        dtf = pd.read_csv("tests/data/dataset.csv.gz")
        feat = dtf.iloc[:, :74]
        out = dtf.drop(dtf.columns[2:74], axis=1)
        # Execute
        fforma_out = loading.fforma.generate_fforma_output(
            models_dict, feat, out
        )
        # Evaluate
        self.assertEqual(fforma_out.shape, (10, 4))
        self.assertEqual(
            fforma_out.columns.to_list(),
            ["dataset", "timeseries", "SHIFT_LR", "SHIFT_SVR"],
        )

    def test_clean_bad_models(self):
        """test loading.fforma.clean_bad_models"""
        # Init
        fforma_out = pd.DataFrame(
            {
                "dataset": [
                    f"dataset{i}" for i in [2, 3, 4, 4, 5, 5, 6, 7, 9]
                ],
                "timeseries": [
                    f"series{i}" for i in [1, 3, 7, 9, 4, 8, 4, 1, 1]
                ],
                "SHIFT_LR": [random.random() for _ in range(9)],
                "SHIFT_SVR": [random.random() for _ in range(9)],
                "MULTI_BAD": [np.inf] * 9,
                "CHAIN_BAD": [np.NAN] * 9,
            }
        )
        # Execute
        loading.fforma.clean_bad_models(fforma_out)
        # Evaluate
        self.assertEqual(fforma_out.shape, (9, 4))
        self.assertEqual(
            fforma_out.columns.to_list(),
            ["dataset", "timeseries", "SHIFT_LR", "SHIFT_SVR"],
        )


class LoadingFFORMASplitFunctionsTest(unittest.TestCase):
    """loading.fforma split functions test"""

    def setUp(self):
        os.makedirs("tests/results")
        datasets = [2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 4]
        series = [1, 2, 3, 1, 2, 3, 4, 1, 2, 3, 4, 5]
        self.fforma_feat = pd.DataFrame(
            {
                "dataset": [f"dataset{i}" for i in datasets],
                "timeseries": [f"series{i}" for i in series],
            }
        )
        for column in CONTEXT_COLUMNS:
            self.fforma_feat[column] = [random.random() for _ in range(12)]
        self.fforma_out = pd.DataFrame(
            {
                "dataset": [f"dataset{i}" for i in datasets],
                "timeseries": [f"series{i}" for i in series],
                "SHIFT_LR": [random.random() for _ in range(12)],
                "SHIFT_SVR": [random.random() for _ in range(12)],
                "MULTI_MLP": [random.random() for _ in range(12)],
                "CHAIN_KNN": [random.random() for _ in range(12)],
            }
        )

    def tearDown(self):
        shutil.rmtree("tests/results")

    def test_split_fforma_in_train_test_ratio0(self):
        """test loading.fforma.split_fforma_in_train_test (ratio 0%)"""
        # Execute
        train_feat, train_out = loading.fforma.split_fforma_in_train_test(
            self.fforma_feat, self.fforma_out, 0, "tests/results"
        )
        # Evaluate
        self.assertEqual(train_feat.shape, (0, len(CONTEXT_COLUMNS) + 2))
        self.assertEqual(train_out.shape, (0, 6))
        self.assertEqual(
            pd.read_csv("tests/results/test_features.csv.gz").shape,
            self.fforma_feat.shape,
        )
        self.assertEqual(
            pd.read_csv("tests/results/test_output.csv.gz").shape,
            self.fforma_out.shape,
        )

    def test_split_fforma_in_train_test_ratio25(self):
        """test loading.fforma.split_fforma_in_train_test (ratio 25%)"""
        # Execute
        train_feat, train_out = loading.fforma.split_fforma_in_train_test(
            self.fforma_feat, self.fforma_out, 0.25, "tests/results"
        )
        # Evaluate
        self.assertEqual(train_feat.shape, (4, len(CONTEXT_COLUMNS) + 2))
        self.assertEqual(train_out.shape, (4, 6))
        self.assertEqual(
            pd.read_csv("tests/results/test_features.csv.gz").shape,
            (8, len(CONTEXT_COLUMNS) + 2),
        )
        self.assertEqual(
            pd.read_csv("tests/results/test_output.csv.gz").shape,
            (8, 6),
        )

    def test_split_fforma_in_train_test_ratio50(self):
        """test loading.fforma.split_fforma_in_train_test (ratio 50%)"""
        # Execute
        train_feat, train_out = loading.fforma.split_fforma_in_train_test(
            self.fforma_feat, self.fforma_out, 0.5, "tests/results"
        )
        # Evaluate
        self.assertEqual(train_feat.shape, (7, len(CONTEXT_COLUMNS) + 2))
        self.assertEqual(train_out.shape, (7, 6))
        self.assertEqual(
            pd.read_csv("tests/results/test_features.csv.gz").shape,
            (5, len(CONTEXT_COLUMNS) + 2),
        )
        self.assertEqual(
            pd.read_csv("tests/results/test_output.csv.gz").shape,
            (5, 6),
        )

    def test_split_fforma_in_train_test_ratio75(self):
        """test loading.fforma.split_fforma_in_train_test (ratio 75%)"""
        # Execute
        train_feat, train_out = loading.fforma.split_fforma_in_train_test(
            self.fforma_feat, self.fforma_out, 0.75, "tests/results"
        )
        # Evaluate
        self.assertEqual(train_feat.shape, (10, len(CONTEXT_COLUMNS) + 2))
        self.assertEqual(train_out.shape, (10, 6))
        self.assertEqual(
            pd.read_csv("tests/results/test_features.csv.gz").shape,
            (2, len(CONTEXT_COLUMNS) + 2),
        )
        self.assertEqual(
            pd.read_csv("tests/results/test_output.csv.gz").shape,
            (2, 6),
        )

    def test_split_fforma_in_train_test_ratio100(self):
        """test loading.fforma.split_fforma_in_train_test (ratio 100%)"""
        # Execute
        train_feat, train_out = loading.fforma.split_fforma_in_train_test(
            self.fforma_feat, self.fforma_out, 1, "tests/results"
        )
        # Evaluate
        self.assertEqual(train_feat.shape, self.fforma_feat.shape)
        self.assertEqual(train_out.shape, self.fforma_out.shape)
        self.assertEqual(
            pd.read_csv("tests/results/test_features.csv.gz").shape,
            (0, len(CONTEXT_COLUMNS) + 2),
        )
        self.assertEqual(
            pd.read_csv("tests/results/test_output.csv.gz").shape,
            (0, 6),
        )
