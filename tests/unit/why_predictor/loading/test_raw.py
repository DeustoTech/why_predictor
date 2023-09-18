"""Unit tests for loading module"""
import unittest
from typing import Any, Dict, List
from unittest.mock import patch

from why_predictor import loading


class LoadingRawDatasetsTest(unittest.TestCase):
    """loading functions test"""

    def setUp(self) -> None:
        self.series = {
            "dataset1": [
                "001.csv.gz",
                "002.csv.gz",
                "003.csv.gz",
                "004.csv.gz",
            ],
            "dataset2": [
                "005.csv.gz",
                "006.csv.gz",
                "007.csv.gz",
                "008.csv.gz",
            ],
        }

    @patch("random.randint")
    def test_select_training_set_0(self, mock_randint: Any) -> None:
        """test select_training_set (0)"""
        mock_randint.side_effect = []
        expected_training_set: Dict[str, List[str]] = {
            "dataset1": [],
            "dataset2": [],
        }
        training_set, testing_set = loading.raw.select_training_set(
            self.series, 0
        )
        self.assertEqual(training_set, expected_training_set)
        self.assertEqual(testing_set, self.series)

    @patch("random.randint")
    def test_select_training_set_25(self, mock_randint: Any) -> None:
        """test select_training_set (25)"""
        mock_randint.side_effect = [0, 2]
        expected_training_set = {
            "dataset1": ["001.csv.gz"],
            "dataset2": ["007.csv.gz"],
        }
        expected_no_training_set = {
            "dataset1": ["002.csv.gz", "003.csv.gz", "004.csv.gz"],
            "dataset2": ["005.csv.gz", "006.csv.gz", "008.csv.gz"],
        }
        training_set, testing_set = loading.raw.select_training_set(
            self.series, 0.25
        )
        self.assertEqual(training_set, expected_training_set)
        self.assertEqual(testing_set, expected_no_training_set)

    @patch("random.randint")
    def test_select_training_set_50(self, mock_randint: Any) -> None:
        """test select_training_set (50)"""
        mock_randint.side_effect = [0, 2, 1, 0]
        expected_training_set = {
            "dataset1": ["001.csv.gz", "004.csv.gz"],
            "dataset2": ["006.csv.gz", "005.csv.gz"],
        }
        expected_no_training_set = {
            "dataset1": ["002.csv.gz", "003.csv.gz"],
            "dataset2": ["007.csv.gz", "008.csv.gz"],
        }
        training_set, testing_set = loading.raw.select_training_set(
            self.series, 0.5
        )
        self.assertEqual(training_set, expected_training_set)
        self.assertEqual(testing_set, expected_no_training_set)

    @patch("random.randint")
    def test_select_training_set_100(self, mock_randint: Any) -> None:
        """test select_training_set (100)"""
        mock_randint.side_effect = [0, 0, 0, 0, 0, 0, 0, 0]
        expected_no_training_set: Dict[str, List[str]] = {
            "dataset1": [],
            "dataset2": [],
        }
        training_set, testing_set = loading.raw.select_training_set(
            self.series, 1
        )
        self.assertEqual(training_set, self.series)
        self.assertEqual(testing_set, expected_no_training_set)

    @patch("random.randint")
    def test_select_training_set_empty(self, mock_randint: Any) -> None:
        """test select_training_set"""
        self.series = {
            "dataset1": [],
            "dataset2": [],
        }
        mock_randint.side_effect = []
        expected_set: Dict[str, List[str]] = {
            "dataset1": [],
            "dataset2": [],
        }
        training_set, testing_set = loading.raw.select_training_set(
            self.series, 0.5
        )
        self.assertEqual(training_set, expected_set)
        self.assertEqual(testing_set, expected_set)

    @patch("random.randint")
    def test_select_training_set_filtered(self, mock_randint: Any) -> None:
        """test select_training_set_filtered"""
        mock_randint.side_effect = [0, 2, 1, 0, 0, 2]
        filtered_files = {"dataset1": ["001.csv.gz", "003.csv.gz"]}
        expected_training_set = {
            "dataset1": ["004.csv.gz", "002.csv.gz"],
            "dataset2": ["005.csv.gz", "008.csv.gz"],
        }
        expected_no_training_set = {
            "dataset1": [],
            "dataset2": ["006.csv.gz", "007.csv.gz"],
        }
        # Execute
        training_set, testing_set = loading.raw.select_training_set_filtered(
            self.series, 0.5, filtered_files
        )
        # Validate
        self.assertEqual(training_set, expected_training_set)
        self.assertEqual(testing_set, expected_no_training_set)

    def test_get_datasets_names(self) -> None:
        """test get_datasets"""
        data_path = "tests/data"
        self.assertCountEqual(
            loading.raw.get_datasets_names(data_path, "imp_csv"),
            ["test1", "test2"],
        )
        self.assertCountEqual(
            loading.raw.get_datasets_names(data_path, "no_exists"), []
        )
