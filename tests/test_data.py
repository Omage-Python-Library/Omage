"""
test_data.py — Real tests for Omage Data pipeline
Uses: Iris, Wine, Breast Cancer CSV files
"""

import unittest
from pathlib import Path

import pandas as pd
import numpy as np

import omage as og
from omage.core.data import OmageDataError

DATA_DIR = Path(__file__).parent


# ================================================================== #
class TestDataLoad(unittest.TestCase):

    def test_load_iris_csv(self):
        data = og.load(str(DATA_DIR / "iris.csv"))
        self.assertIsInstance(data, og.Data)
        self.assertEqual(len(data), 150)

    def test_load_wine_csv(self):
        data = og.load(str(DATA_DIR / "wine.csv"))
        self.assertEqual(len(data), 178)

    def test_load_breast_cancer_csv(self):
        data = og.load(str(DATA_DIR / "breast_cancer.csv"))
        self.assertEqual(len(data), 569)

    def test_load_missing_file_raises(self):
        with self.assertRaises(OmageDataError):
            og.load("nonexistent_file.csv")

    def test_load_wrong_extension_raises(self):
        with self.assertRaises(OmageDataError):
            og.load("model.omg")

    def test_repr(self):
        data = og.load(str(DATA_DIR / "iris.csv"))
        self.assertIn("150", repr(data))


# ================================================================== #
class TestDataTarget(unittest.TestCase):

    def test_set_target_valid(self):
        data = og.load(str(DATA_DIR / "iris.csv"))
        data.set_target("target")
        self.assertEqual(data._target_col, "target")

    def test_set_target_invalid_column(self):
        data = og.load(str(DATA_DIR / "iris.csv"))
        with self.assertRaises(OmageDataError):
            data.set_target("nonexistent_column")

    def test_target_excluded_from_features(self):
        data = og.load(str(DATA_DIR / "iris.csv"))
        data.set_target("target").split(0.8)
        loader = data.get_train_loader()
        for X, y in loader:
            # Iris has 4 features — target column excluded
            self.assertEqual(X.shape[1], 4)
            break


# ================================================================== #
class TestDataClean(unittest.TestCase):

    def test_clean_removes_nulls(self):
        df = pd.DataFrame({
            "a": [1.0, None, 3.0, 4.0],
            "b": [2.0, 3.0, 4.0, 5.0],
            "target": [0, 1, 0, 1],
        })
        data = og.Data(data=df)
        data.clean()
        self.assertEqual(len(data), 3)

    def test_clean_removes_duplicates(self):
        df = pd.DataFrame({
            "a":      [1.0, 1.0, 2.0],
            "target": [0,   0,   1],
        })
        data = og.Data(data=df)
        data.clean()
        self.assertEqual(len(data), 2)

    def test_clean_on_iris_preserves_all_rows(self):
        data = og.load(str(DATA_DIR / "iris.csv"))
        data.clean()
        self.assertGreaterEqual(len(data), 149)   # Iris may have 1 duplicate

    def test_clean_on_empty_data_raises(self):
        data = og.Data()
        with self.assertRaises(OmageDataError):
            data.clean()


# ================================================================== #
class TestDataNormalize(unittest.TestCase):

    def test_normalize_range(self):
        data = og.load(str(DATA_DIR / "iris.csv"))
        data.set_target("target").clean().normalize(0, 1)
        numeric = data.data.select_dtypes(include=[np.number])
        feature_cols = [c for c in numeric.columns if c != "target"]
        for col in feature_cols:
            self.assertGreaterEqual(data.data[col].min(), -0.01)
            self.assertLessEqual(data.data[col].max(),     1.01)

    def test_normalize_custom_range(self):
        data = og.load(str(DATA_DIR / "iris.csv"))
        data.set_target("target").normalize(-1, 1)
        feature_cols = [c for c in data.data.columns if c != "target"]
        for col in feature_cols:
            self.assertGreaterEqual(data.data[col].min(), -1.01)
            self.assertLessEqual(data.data[col].max(),     1.01)

    def test_normalize_does_not_change_target(self):
        data = og.load(str(DATA_DIR / "iris.csv"))
        original_targets = data.data["target"].unique().tolist()
        data.set_target("target").normalize(0, 1)
        new_targets = data.data["target"].unique().tolist()
        self.assertEqual(sorted(original_targets), sorted(new_targets))


# ================================================================== #
class TestDataSplit(unittest.TestCase):

    def test_split_80_20(self):
        data = og.load(str(DATA_DIR / "iris.csv"))
        data.split(0.8)
        self.assertEqual(len(data.train_data), 120)
        self.assertEqual(len(data.test_data),   30)

    def test_split_70_30(self):
        data = og.load(str(DATA_DIR / "wine.csv"))
        data.split(0.7)
        total = len(data.train_data) + len(data.test_data)
        self.assertEqual(total, 178)

    def test_split_invalid_ratio_raises(self):
        data = og.load(str(DATA_DIR / "iris.csv"))
        with self.assertRaises(OmageDataError):
            data.split(1.5)

    def test_split_creates_loaders(self):
        data = og.load(str(DATA_DIR / "iris.csv"))
        data.set_target("target").split(0.8)
        train_loader = data.get_train_loader()
        test_loader  = data.get_test_loader()
        self.assertIsNotNone(train_loader)
        self.assertIsNotNone(test_loader)


# ================================================================== #
class TestDataShuffle(unittest.TestCase):

    def test_shuffle_same_length(self):
        data = og.load(str(DATA_DIR / "iris.csv"))
        data.shuffle(seed=42)
        self.assertEqual(len(data), 150)

    def test_shuffle_with_seed_reproducible(self):
        data1 = og.load(str(DATA_DIR / "iris.csv"))
        data2 = og.load(str(DATA_DIR / "iris.csv"))
        data1.shuffle(seed=0)
        data2.shuffle(seed=0)
        self.assertTrue(data1.data.equals(data2.data))

    def test_shuffle_different_seeds_differ(self):
        data1 = og.load(str(DATA_DIR / "iris.csv"))
        data2 = og.load(str(DATA_DIR / "iris.csv"))
        data1.shuffle(seed=1)
        data2.shuffle(seed=99)
        self.assertFalse(data1.data.equals(data2.data))


# ================================================================== #
class TestDataBatch(unittest.TestCase):

    def test_batch_size_set(self):
        data = og.load(str(DATA_DIR / "iris.csv"))
        data.set_target("target").batch(16).split(0.8)
        loader = data.get_train_loader()
        for X, y in loader:
            self.assertLessEqual(X.shape[0], 16)
            break

    def test_invalid_batch_raises(self):
        data = og.load(str(DATA_DIR / "iris.csv"))
        with self.assertRaises(OmageDataError):
            data.batch(-1)


# ================================================================== #
class TestDataChaining(unittest.TestCase):

    def test_full_chain_iris(self):
        """Full pipeline: load → clean → normalize → shuffle → split → loaders"""
        data = (
            og.load(str(DATA_DIR / "iris.csv"))
              .set_target("target")
              .clean()
              .normalize(0, 1)
              .shuffle(seed=7)
              .batch(32)
              .split(0.8)
        )
        self.assertIsNotNone(data.train_data)
        self.assertIsNotNone(data.test_data)
        train_loader = data.get_train_loader()
        for X, y in train_loader:
            self.assertEqual(X.shape[1], 4)
            self.assertIn(y[0].item(), [0, 1, 2])
            break

    def test_full_chain_breast_cancer(self):
        """Binary classification pipeline"""
        data = (
            og.load(str(DATA_DIR / "breast_cancer.csv"))
              .set_target("target")
              .clean()
              .normalize()
              .shuffle(seed=42)
              .split(0.8)
        )
        loader = data.get_train_loader()
        for X, y in loader:
            self.assertEqual(X.shape[1], 30)
            self.assertIn(y[0].item(), [0, 1])
            break


if __name__ == "__main__":
    unittest.main(verbosity=2)
