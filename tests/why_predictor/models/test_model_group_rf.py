"""Test for RF model group"""
import json
import os
import shutil
import unittest
from glob import glob
from typing import Tuple

import pandas as pd  # type: ignore

from why_predictor.errors import ErrorType
from why_predictor.models import ModelGroups


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


class TestRFModelGroup(unittest.TestCase):
    """Tests for Multi_RF ModelGroup."""

    def setUp(self):
        self.model_name = "RF"
        self.combinations = 9
        self.base_path = "tests/results"
        self.feat, self.out = generate_dirs_and_files_for_testing()
        self.group = None

    def _set_up(self):
        """just common code to set up the tests"""
        self.group = ModelGroups[self.model_name].value(
            self.model_name,
            (self.feat.iloc[:50], self.out.iloc[:50]),
            ErrorType.MAPE2,
            self.base_path,
        )

    def tearDown(self):
        shutil.rmtree(self.base_path)

    def _test_create_group(self):
        """test create object"""
        # Evaluate
        self.assertEqual(len(self.group.hyper_params), self.combinations)

    def _test_fit(self, error_value):
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

    def _test_fit_from_files(self, error_value):
        """test fit hyperparams from files"""
        # Execute
        self.group.fit_from_files()
        # Validate
        self.__validate_fit(error_value)

    def __validate_fit(self, error_value):
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
        self.assertAlmostEqual(error, error_value, delta=0.1)
        self.assertEqual(
            set(hyperparameters.keys()), {"n_estimators", "max_depth"}
        )


class TestMultiRFModelGroup(TestRFModelGroup):
    """Tests for Multi_RF ModelGroup."""

    def setUp(self):
        super().setUp()
        self.model_name = "MULTI_RF"
        self.multi_rf_error = 0.47723960876464844
        self._set_up()

    def test_create_group(self):
        """test create object"""
        self._test_create_group()

    def test_fit(self):
        """test fit hyperparams"""
        self._test_fit(self.multi_rf_error)

    def test_fit_from_files(self):
        """test fit hyperparams from files"""
        self._test_fit_from_files(self.multi_rf_error)


class TestChainRFModelGroup(TestRFModelGroup):
    """Tests for Chain_RF ModelGroup."""

    def setUp(self):
        super().setUp()
        self.model_name = "CHAIN_RF"
        self.chain_rf_error = 0.5703571438789368
        self._set_up()

    def test_create_group(self):
        """test create object"""
        self._test_create_group()

    def test_fit(self):
        """test fit hyperparams"""
        self._test_fit(self.chain_rf_error)

    def test_fit_from_files(self):
        """test fit hyperparams from files"""
        self._test_fit_from_files(self.chain_rf_error)


class TestShiftRFModelGroup(TestRFModelGroup):
    """Tests for Shift_RF ModelGroup."""

    def setUp(self):
        super().setUp()
        self.model_name = "SHIFT_RF"
        self.shift_rf_error = 0.5112500190734863
        self._set_up()

    def test_create_group(self):
        """test create object"""
        self._test_create_group()

    def test_fit(self):
        """test fit hyperparams"""
        self._test_fit(self.shift_rf_error)

    def test_fit_from_files(self):
        """test fit hyperparams from files"""
        self._test_fit_from_files(self.shift_rf_error)
