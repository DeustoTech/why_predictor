"""Test for KNN model group"""
import json
import os
import shutil
import unittest
from glob import glob
from typing import Tuple

import pandas as pd  # type: ignore

from why_predictor.errors import ErrorType
from why_predictor.models import ModelGroups
from why_predictor.models.model_group import BasicModelGroup


def generate_dirs_and_files_for_testing() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Generate the needed files and directories for testing"""
    num_features = 72
    os.makedirs("tests/results/models")
    os.makedirs("tests/results/hyperparameters")
    os.makedirs("tests/results/predictions")
    os.makedirs("tests/results/errors/sum")
    os.makedirs("tests/results/errors/raw")
    os.makedirs("tests/results/test/test1/output")
    dtf = pd.read_csv("tests/data/rolling.csv.gz")
    feat = dtf.iloc[:, : num_features + 2]
    out = dtf.drop(dtf.iloc[:, 2 : num_features + 2], axis=1)
    index1 = feat.shape[0] - 4
    index2 = feat.shape[0] - 2
    # Change name and timeseries (training) - Features
    feat.iloc[:25] = feat.iloc[:25].replace("mydataset", "test1")
    feat.iloc[:25] = feat.iloc[:25].replace("mytimeseries", "series1")
    feat.iloc[25:50] = feat.iloc[25:50].replace("mydataset", "test2")
    feat.iloc[25:50] = feat.iloc[25:50].replace("mytimeseries", "series2")
    # Change name and timeseries (training) - Output
    out.iloc[:25] = out.iloc[:25].replace("mydataset", "test1")
    out.iloc[:25] = out.iloc[:25].replace("mytimeseries", "series1")
    out.iloc[25:50] = out.iloc[25:50].replace("mydataset", "test2")
    out.iloc[25:50] = out.iloc[25:50].replace("mytimeseries", "series2")
    # - Save features dtf 1
    os.makedirs(
        "tests/results/test/test1/features",
    )
    (
        feat.iloc[index1:index2]
        .replace("mydataset", "test1")
        .replace("mytimeseries", "series1")
        .to_csv(
            "tests/results/test/test1/features/series1.csv.gz",
            index=False,
        )
    )
    # - Save output dtf 1
    os.makedirs("tests/results/test/test1/output", exist_ok=True)
    (
        out.iloc[index1:index2]
        .replace("mydataset", "test1")
        .replace("mytimeseries", "series1")
        .to_csv(
            "tests/results/test/test1/output/series1.csv.gz",
            index=False,
        )
    )
    # - Save features dtf 2
    os.makedirs("tests/results/test/test2/features", exist_ok=True)
    (
        feat.iloc[index2:]
        .replace("mydataset", "test2")
        .replace("mytimeseries", "series2")
        .to_csv(
            "tests/results/test/test2/features/series2.csv.gz",
            index=False,
        )
    )
    # - Save output dtf 2
    os.makedirs("tests/results/test/test2/output", exist_ok=True)
    (
        out.iloc[index2:]
        .replace("mydataset", "test2")
        .replace("mytimeseries", "series2")
        .to_csv(
            "tests/results/test/test2/output/series2.csv.gz",
            index=False,
        )
    )
    # Return feat & out
    return feat, out


class TestKNNModelGroup(unittest.TestCase):
    """Base tests for KNN ModelGroup."""

    def setUp(self) -> None:
        self.model_name = "KNN"
        self.base_path = "tests/results"
        self.combinations = 3
        self.feat, self.out = generate_dirs_and_files_for_testing()
        self.group: BasicModelGroup

    def _set_up(self) -> BasicModelGroup:
        group: BasicModelGroup = ModelGroups[self.model_name].value(
            self.model_name,
            (self.feat.iloc[:50], self.out.iloc[:50]),
            ErrorType.MAPE2,
            self.base_path,
        )
        return group

    def tearDown(self) -> None:
        shutil.rmtree(self.base_path)

    def _test_create_group(self) -> None:
        """test create object"""
        # Evaluate
        self.assertEqual(len(self.group.hyper_params), self.combinations)

    def _test_fit(self, error_value: float) -> None:
        """test fit hyperparams"""
        # Init
        index = self.feat.shape[0] - 4
        # Execute
        self.group.fit(
            self.feat.iloc[index:].reset_index(drop=True),
            self.out.iloc[index:].reset_index(drop=True),
        )
        # Validate
        self.__validate_fit(error_value)

    def _test_fit_from_files(self, error_value: float) -> None:
        """test fit hyperparams from files"""
        # Execute
        self.group.fit_from_files()
        # Validate
        self.__validate_fit(error_value)

    def __validate_fit(self, error_value: float) -> None:
        # Validate
        # - hyperparams error files
        self.assertEqual(
            len(glob(os.path.join(self.base_path, "errors", "sum", "*"))),
            self.combinations,
        )
        # - best hyperparam file
        self.assertEqual(
            len(glob(os.path.join(self.base_path, "hyperparameters", "*"))), 1
        )
        # - models
        self.assertEqual(
            len(glob(os.path.join(self.base_path, "models", "*"))),
            self.combinations,
        )
        # - hyperparameters result
        filename = os.path.join(
            self.base_path, "hyperparameters", f"{self.model_name}.json"
        )
        with open(filename, encoding="utf8") as fhyper:
            error_text, hyper_text = fhyper.read().split("|")
            error = float(error_text)
            hyperparameters = json.loads(hyper_text)
        self.assertEqual(error, error_value)
        self.assertEqual(
            hyperparameters, {"n_neighbors": 5, "weights": "distance"}
        )


class TestMultiKNNModelGroup(TestKNNModelGroup):
    """Tests for Multi_KNN ModelGroup."""

    def setUp(self) -> None:
        super().setUp()
        self.model_name = "MULTI_KNN"
        self.group = self._set_up()

    def test_create_group(self) -> None:
        """test create object"""
        self._test_create_group()

    def test_fit(self) -> None:
        """test fit hyperparams"""
        self._test_fit(0.34283000230789185)

    def test_fit_from_files(self) -> None:
        """test fit hyperparams from files"""
        self._test_fit_from_files(0.34283000230789185)


class TestChainKNNModelGroup(TestKNNModelGroup):
    """Tests for Chain_KNN ModelGroup."""

    def setUp(self) -> None:
        super().setUp()
        self.model_name = "CHAIN_KNN"
        self.group = self._set_up()

    def test_create_group(self) -> None:
        """test create object"""
        self._test_create_group()

    def test_fit(self) -> None:
        """test fit hyperparams"""
        self._test_fit(0.33348098397254944)

    def test_fit_from_files(self) -> None:
        """test fit hyperparams from files"""
        self._test_fit_from_files(0.33348098397254944)


class TestShiftKNNModelGroup(TestKNNModelGroup):
    """Tests for Shift_KNN ModelGroup."""

    def setUp(self) -> None:
        super().setUp()
        self.model_name = "SHIFT_KNN"
        self.group = self._set_up()

    def test_create_group(self) -> None:
        """test create object"""
        self._test_create_group()

    def test_fit(self) -> None:
        """test fit hyperparams"""
        self._test_fit(0.36563628911972046)

    def test_fit_from_files(self) -> None:
        """test fit hyperparams from files"""
        self._test_fit_from_files(0.36563628911972046)
