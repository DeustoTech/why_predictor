"""Tests for phase 2"""
import glob
import os
import shutil
import unittest
from unittest.mock import patch

import pandas as pd  # type: ignore

from why_predictor.phases import phase3


class TestPhase3Functions(unittest.TestCase):
    """tests phase 3"""

    def setUp(self) -> None:
        self.base_path = "tests/results"
        os.makedirs(self.base_path)

    def tearDown(self) -> None:
        shutil.rmtree(self.base_path, ignore_errors=True)

    @patch("why_predictor.config.TRAINING_PATH", "tests/results")
    def test_delete_previous_execution(self) -> None:
        """test delete_previous_execution (phase3)"""
        phase3_path = self.base_path
        os.makedirs(phase3_path, exist_ok=True)
        self.assertTrue(os.path.exists(phase3_path))
        phase3.delete_previous_execution(phase3_path)
        self.assertFalse(os.path.exists(phase3_path))

    @patch("why_predictor.config.NUM_FEATURES", 66)
    @patch("why_predictor.config.NUM_PREDICTIONS", 4)
    def test_generate_phase2_tree(self) -> None:
        """test generate_phase2_tree"""
        num_features = 66
        num_predictions = 4
        # Execute
        phase3.generate_phase3_tree(self.base_path)
        # Validate
        phase2_path = os.path.join(self.base_path, "phase2")
        self.assertTrue(os.path.exists(phase2_path))
        paths_to_check = [
            ["hyperparameters"],
            ["models"],
            ["post-hoc"],
            ["errors", "sum"],
            ["errors", "raw"],
            ["datasets", "test"],
            ["datasets", "train"],
        ]
        for path in paths_to_check:
            self.assertTrue(os.path.exists(os.path.join(phase2_path, *path)))
        # - phase1
        phase1_path = os.path.join(phase2_path, "phase1")
        self.assertTrue(os.path.exists(phase1_path))
        paths_to_check = [
            ["hyperparameters"],
            ["models"],
            ["post-hoc"],
            ["errors", "sum"],
            ["errors", "raw"],
            ["datasets", "test"],
            ["datasets", "train"],
        ]
        for path in paths_to_check:
            self.assertTrue(os.path.exists(os.path.join(phase1_path, *path)))
        # - phase1 datasets
        self.assertTrue(
            os.path.exists(os.path.join(phase1_path, "train_features.csv.gz"))
        )
        self.assertTrue(
            os.path.exists(os.path.join(phase1_path, "train_output.csv.gz"))
        )
        self.assertTrue(
            os.path.exists(os.path.join(phase1_path, "test_features.csv.gz"))
        )
        self.assertTrue(
            os.path.exists(os.path.join(phase1_path, "test_output.csv.gz"))
        )
        df_train_features = pd.read_csv(
            os.path.join(phase1_path, "train_features.csv.gz")
        )
        df_train_output = pd.read_csv(
            os.path.join(phase1_path, "train_output.csv.gz")
        )
        df_test_features = pd.read_csv(
            os.path.join(phase1_path, "test_features.csv.gz")
        )
        df_test_output = pd.read_csv(
            os.path.join(phase1_path, "test_output.csv.gz")
        )
        self.assertEqual(len(df_train_features.columns), num_features + 2)
        self.assertEqual(len(df_train_output.columns), num_predictions + 2)
        self.assertEqual(len(df_test_features.columns), num_features + 2)
        self.assertEqual(len(df_test_output.columns), num_predictions + 2)
        self.assertEqual(len(df_train_features), 0)
        self.assertEqual(len(df_train_output), 0)
        self.assertEqual(len(df_test_features), 0)
        self.assertEqual(len(df_test_output), 0)

    def test_move_models_to_phase3(self) -> None:
        """test phase3.move_models_to_phase3"""
        # Init
        phase2_path = os.path.join(self.base_path, "phase2")
        for directory in ["models", "hyperparameters"]:
            os.makedirs(os.path.join(phase2_path, directory))
        phase1_path = os.path.join(phase2_path, "phase1")
        for directory in ["models", "hyperparameters"]:
            os.makedirs(os.path.join(phase1_path, directory))
        # Execute
        phase3.move_models_to_phase3(
            self.base_path, "tests/data", "tests/data"
        )
        # Evaluate
        # - phase2
        for directory in ["hyperparameters", "models"]:
            fforma_eval_path = os.path.join(phase2_path, directory, "*")
            fforma_path = os.path.join("tests/data", directory, "*")
            self.assertEqual(
                [os.path.split(x)[-1] for x in glob.glob(fforma_eval_path)],
                [os.path.split(x)[-1] for x in glob.glob(fforma_path)],
            )
            self.assertEqual(len(glob.glob(fforma_eval_path)), 2)
        # - phase1
        for directory in ["hyperparameters", "models"]:
            fforma_path = os.path.join(phase1_path, directory, "*")
            training_path = os.path.join("tests/data", directory, "*")
            self.assertEqual(
                [os.path.split(x)[-1] for x in glob.glob(fforma_path)],
                [os.path.split(x)[-1] for x in glob.glob(training_path)],
            )
            self.assertEqual(len(glob.glob(fforma_path)), 2)
