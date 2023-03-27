"""Test for Linear regression"""
import json
import os
import shutil
import unittest
from typing import Any, Dict

import pandas as pd  # type: ignore

from why_predictor.models import linear_regression as lr

NAME = "Chained Linear Regression"
SHORT_NAME = "CHAIN_LR"


class TestChainedLinearRegressorBasic(unittest.TestCase):
    """Tests for ChainedLinearRegressor (Basic info)"""

    def setUp(self) -> None:
        num_features = 72
        os.makedirs("tests/results/models")
        os.makedirs("tests/results/hyperparameters")
        os.makedirs("tests/results/predictions")
        os.makedirs("tests/results/errors/sum")
        os.makedirs("tests/results/errors/raw")
        dtf = pd.read_csv("tests/data/rolling.csv.gz")
        self.feat = dtf.iloc[:, : num_features + 2]
        self.out = dtf.drop(dtf.iloc[:, 2 : num_features + 2], axis=1)
        self.hyperparams: Dict[str, Any] = {}
        self.model = lr.ChainedLinearRegressor(
            self.hyperparams, "tests/results"
        )

    def tearDown(self) -> None:
        shutil.rmtree("tests/results")

    def test_hyperparams(self) -> None:
        """Test .hyperparams"""
        self.assertEqual(self.model.hyperparams, self.hyperparams)

    def test_paramsname(self) -> None:
        """Test .params name"""
        self.assertEqual(
            self.model.paramsname,
            f"{SHORT_NAME}_{json.dumps(self.hyperparams)}",
        )

    def test_name(self) -> None:
        """Test .name"""
        self.assertEqual(self.model.name, NAME)

    def test_short_name(self) -> None:
        """Test .short_name"""
        self.assertEqual(self.model.short_name, SHORT_NAME)

    def test_path(self) -> None:
        """Test .path"""
        self.assertEqual(
            self.model.path,
            "tests/results/models/"
            + f"{SHORT_NAME}_{json.dumps(self.hyperparams)}",
        )

    def test_load_model_error(self) -> None:
        """Test .load_model"""
        self.assertRaises(FileNotFoundError, self.model.load_model)
