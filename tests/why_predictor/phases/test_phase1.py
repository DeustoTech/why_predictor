"""Tests for phase 1"""
import os
import shutil
import unittest
from unittest.mock import patch

import pandas as pd  # type: ignore

from why_predictor.phases import phase1


class TestPhase1Functions(unittest.TestCase):
    """tests phase 1"""

    def setUp(self) -> None:
        self.base_path = "tests/results"
        os.makedirs(self.base_path)

    def tearDown(self) -> None:
        if os.path.exists(self.base_path):
            shutil.rmtree(self.base_path)

    @patch("why_predictor.config.TRAINING_PATH", "tests/results")
    def test_delete_previous_execution(self) -> None:
        """test delete_previous_execution"""
        os.makedirs(self.base_path, exist_ok=True)
        self.assertTrue(os.path.exists(self.base_path))
        phase1.delete_previous_execution(self.base_path)
        self.assertFalse(os.path.exists(self.base_path))

    @patch("why_predictor.config.NUM_FEATURES", 66)
    @patch("why_predictor.config.NUM_PREDICTIONS", 4)
    def test_generate_phase1_tree(self) -> None:
        """test generate_phase1_tree"""
        num_features = 66
        num_predictions = 4
        # Execute
        phase1.generate_phase1_tree(self.base_path)
        # Validate
        self.assertTrue(os.path.exists(os.path.join(self.base_path, "models")))
        self.assertTrue(
            os.path.exists(os.path.join(self.base_path, "hyperparameters"))
        )
        self.assertTrue(
            os.path.exists(os.path.join(self.base_path, "predictions"))
        )
        self.assertTrue(
            os.path.exists(os.path.join(self.base_path, "post-hoc"))
        )
        self.assertTrue(
            os.path.exists(os.path.join(self.base_path, "errors", "raw"))
        )
        self.assertTrue(
            os.path.exists(os.path.join(self.base_path, "errors", "sum"))
        )
        self.assertTrue(
            os.path.exists(os.path.join(self.base_path, "datasets", "train"))
        )
        self.assertTrue(
            os.path.exists(os.path.join(self.base_path, "datasets", "test"))
        )
        self.assertTrue(
            os.path.exists(
                os.path.join(self.base_path, "train_features.csv.gz")
            )
        )
        self.assertTrue(
            os.path.exists(os.path.join(self.base_path, "train_output.csv.gz"))
        )
        self.assertTrue(
            os.path.exists(
                os.path.join(self.base_path, "test_features.csv.gz")
            )
        )
        self.assertTrue(
            os.path.exists(os.path.join(self.base_path, "test_output.csv.gz"))
        )
        df_train_features = pd.read_csv(
            os.path.join(self.base_path, "train_features.csv.gz")
        )
        df_train_output = pd.read_csv(
            os.path.join(self.base_path, "train_output.csv.gz")
        )
        df_test_features = pd.read_csv(
            os.path.join(self.base_path, "test_features.csv.gz")
        )
        df_test_output = pd.read_csv(
            os.path.join(self.base_path, "test_output.csv.gz")
        )
        self.assertEqual(len(df_train_features.columns), num_features + 2)
        self.assertEqual(len(df_train_output.columns), num_predictions + 2)
        self.assertEqual(len(df_test_features.columns), num_features + 2)
        self.assertEqual(len(df_test_output.columns), num_predictions + 2)
        self.assertEqual(len(df_train_features), 0)
        self.assertEqual(len(df_train_output), 0)
        self.assertEqual(len(df_test_features), 0)
        self.assertEqual(len(df_test_output), 0)
