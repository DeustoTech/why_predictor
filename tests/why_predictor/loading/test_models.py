"""Test loading.models functions"""
import os
import shutil
import unittest
from unittest.mock import Mock, patch

import pandas as pd  # type: ignore

from why_predictor import loading
from why_predictor.models import BasicModel
from why_predictor.models.linear_regression import ShiftedLinearRegressor


class BasicModelClassMock(ShiftedLinearRegressor):
    """ShiftedLinearRegressor Mocked Class for testing examples of BasicModel
    objects"""

    def __init__(self, hyperparams, base_path):
        super().__init__(hyperparams, base_path)
        self.generate_model = Mock()


class TestLoadSetsModels(unittest.TestCase):
    """load_set.models test"""

    def setUp(self):
        os.makedirs("tests/results/")
        shutil.copytree("tests/data/models", "tests/results/models")
        shutil.copytree(
            "tests/data/hyperparameters", "tests/results/hyperparameters"
        )
        # BasicModelClassMock.set_up_mock()

    def tearDown(self):
        shutil.rmtree("tests/results")

    def test_load_best_trained_models(self):
        """load_best_trained_models test"""
        dict_models = loading.models.load_best_trained_models("tests/results")
        self.assertTrue(list(dict_models.keys()), ["SHIFT_LR", "SHIFT_SVR"])
        for model in dict_models.values():
            self.assertTrue(isinstance(model, BasicModel))

    @patch("why_predictor.loading.models.load_best_trained_models")
    def test_load_or_train_best_models_load(self, mocked_method):
        """load_or_train_best_models test (load)"""
        # Init
        mocked_method.side_effect = [
            {
                "SHIFT_LR": BasicModelClassMock({}, "tests/results"),
                "SHIFT_SVR": BasicModelClassMock({}, "tests/results"),
            }
        ]
        datasets = [[], []]
        # Execute
        dict_models = loading.models.load_or_train_best_models(
            datasets, "tests/results"
        )
        # Validate
        self.assertTrue(list(dict_models.keys()), ["SHIFT_LR", "SHIFT_SVR"])
        for model in dict_models.values():
            self.assertTrue(isinstance(model, BasicModel))
            model.generate_model.assert_not_called()

    @patch("why_predictor.loading.models.load_best_trained_models")
    @patch.object(BasicModel, "save_model")
    def test_load_or_train_best_models_load_train(self, mocked_function, _):
        """load_or_train_best_models test (train)"""
        # Init
        mocked_function.side_effect = [
            {
                "SHIFT_LR": BasicModelClassMock({}, "tests/results"),
                "SHIFT_SVR": BasicModelClassMock({}, "tests/results"),
            }
        ]
        dtf = pd.read_csv("tests/data/rolling2.csv.gz")
        datasets = [dtf.iloc[:5, :74], dtf.iloc[:5, 74:]]
        # Execute
        dict_models = loading.models.load_or_train_best_models(
            datasets, "tests/results"
        )
        # Validate
        for model in dict_models.values():
            model.generate_model.assert_called_once()

    def test_load_best_trained_model_from_list(self):
        """load_best_trained_model_from_list test"""
        # Init
        dtf = pd.read_csv("tests/data/rolling2.csv.gz")
        datasets = [dtf.iloc[:5, :74], dtf.iloc[:5, 74:]]
        # Execute
        model = loading.models.load_or_train_best_trained_model(
            datasets, "tests/results"
        )
        # Validate
        self.assertTrue(model.name, "SHIFT_LR")
        self.assertTrue(isinstance(model, BasicModel))

    def test_load_error_and_hyperparameters(self):
        """load_best_trained_model test"""
        filename = "tests/results/hyperparameters/SHIFT_SVR.json"
        error, hyperparams = loading.models.load_error_and_hyperparameters(
            filename
        )
        self.assertEqual(error, 0.941211462020874)
        self.assertEqual(
            hyperparams,
            {
                "epsilon": 0.0,
                "tol": 0.0001,
                "C": 1.0,
                "loss": "epsilon_insensitive",
            },
        )
