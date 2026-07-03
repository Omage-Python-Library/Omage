"""
test_model.py — Real tests for Omage Model
Uses: Iris (3-class), Wine (3-class), Breast Cancer (binary)
"""

import os
import tempfile
import unittest
from pathlib import Path

import omage as og
from omage.core.model import OmageModelError

DATA_DIR = Path(__file__).parent


def make_classifier(num_classes: int = 3):
    return og.model(
        type="classifier",
        layers=[
            og.dense(64),
            og.dropout(0.2),
            og.dense(32),
            og.dense(num_classes, activation="softmax"),
        ],
        optimizer=og.adam(lr=0.01),
        loss=og.cross_entropy(),
    )


def load_iris() -> og.Data:
    data = og.load(str(DATA_DIR / "iris.csv"))
    data.set_target("target").clean().normalize().shuffle(seed=42).split(0.8)
    return data


def load_wine() -> og.Data:
    data = og.load(str(DATA_DIR / "wine.csv"))
    data.set_target("target").clean().normalize().shuffle(seed=42).split(0.8)
    return data


def load_bc() -> og.Data:
    data = og.load(str(DATA_DIR / "breast_cancer.csv"))
    data.set_target("target").clean().normalize().shuffle(seed=42).split(0.8)
    return data


# ================================================================== #
class TestModelCreation(unittest.TestCase):

    def test_create_classifier(self):
        m = make_classifier()
        self.assertEqual(m.type, "classifier")
        self.assertFalse(m.trained)
        self.assertIsNotNone(m.torch_model)

    def test_invalid_type_raises(self):
        with self.assertRaises(OmageModelError):
            og.model(type="invalid_type", layers=[og.dense(10)])

    def test_no_layers_raises(self):
        with self.assertRaises(OmageModelError):
            og.model(type="classifier", layers=[])

    def test_repr(self):
        m = make_classifier()
        self.assertIn("classifier", repr(m))
        self.assertIn("trained=False", repr(m))


# ================================================================== #
class TestModelTrainIris(unittest.TestCase):

    def setUp(self):
        """Fresh model and data per test"""
        self.data  = load_iris()
        self.model = make_classifier(num_classes=3)

    def test_train_returns_history(self):
        history = self.model.train(self.data, epochs=5, verbose=False)
        self.assertIn("loss",     history)
        self.assertIn("accuracy", history)
        self.assertEqual(len(history["loss"]), 5)

    def test_loss_decreases(self):
        history = self.model.train(self.data, epochs=10, verbose=False)
        self.assertLess(history["loss"][-1], history["loss"][0])

    def test_accuracy_above_threshold(self):
        self.model.train(self.data, epochs=20, verbose=False)
        metrics = self.model.evaluate(self.data)
        self.assertGreater(metrics["accuracy"], 50.0)

    def test_trained_flag(self):
        self.model.train(self.data, epochs=3, verbose=False)
        self.assertTrue(self.model.trained)

    def test_history_structure(self):
        history = self.model.train(self.data, epochs=5, verbose=False)
        self.assertIsInstance(history["loss"],     list)
        self.assertIsInstance(history["accuracy"], list)
        for loss_val in history["loss"]:
            self.assertGreater(loss_val, 0)


# ================================================================== #
class TestModelTrainWine(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.data  = load_wine()
        cls.model = make_classifier(num_classes=3)
        cls.model.train(cls.data, epochs=15, verbose=False)

    def test_evaluate_returns_metrics(self):
        metrics = self.model.evaluate(self.data)
        self.assertIn("accuracy", metrics)
        self.assertIn("loss",     metrics)

    def test_accuracy_positive(self):
        metrics = self.model.evaluate(self.data)
        self.assertGreaterEqual(metrics["accuracy"], 0.0)
        self.assertLessEqual(metrics["accuracy"],    100.0)

    def test_loss_positive(self):
        metrics = self.model.evaluate(self.data)
        self.assertGreater(metrics["loss"], 0.0)


# ================================================================== #
class TestModelTrainBreastCancer(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.data  = load_bc()
        cls.model = og.model(
            type="classifier",
            layers=[
                og.dense(128),
                og.dropout(0.3),
                og.dense(64),
                og.dropout(0.2),
                og.dense(2, activation="softmax"),
            ],
            optimizer=og.adam(lr=0.001),
            loss=og.cross_entropy(),
        )
        cls.model.train(cls.data, epochs=20, verbose=False)

    def test_binary_accuracy_above_70(self):
        metrics = self.model.evaluate(self.data)
        self.assertGreater(metrics["accuracy"], 70.0,
            f"Expected >70% on breast cancer, got {metrics['accuracy']:.2f}%")

    def test_predict_returns_result(self):
        import torch
        sample = torch.randn(1, 30)
        result = self.model.predict(sample)
        self.assertIsNotNone(result)


# ================================================================== #
class TestModelSaveLoad(unittest.TestCase):

    def test_save_and_load_weights(self):
        data  = load_iris()
        model = make_classifier(num_classes=3)
        model.train(data, epochs=5, verbose=False)

        with tempfile.NamedTemporaryFile(suffix=".omg", delete=False) as f:
            path = f.name
        try:
            model.save(path)
            self.assertTrue(os.path.exists(path))
            self.assertGreater(os.path.getsize(path), 0)

            model2 = make_classifier(num_classes=3)
            import torch
            model2.torch_model(torch.randn(1, 4))
            model2.load(path)
            self.assertTrue(model2.trained)
        finally:
            os.unlink(path)

    def test_checkpoint_created(self):
        data  = load_iris()
        model = make_classifier(num_classes=3)
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt = os.path.join(tmpdir, "checkpoint.omg")
            model.train(data, epochs=3, verbose=False,
                        checkpoint_path=ckpt, checkpoint_every=1)
            saved = list(Path(tmpdir).glob("*.omg"))
            self.assertGreater(len(saved), 0)


# ================================================================== #
class TestModelCallbacks(unittest.TestCase):

    def test_on_train_callback(self):
        called = []
        data  = load_iris()
        model = make_classifier(num_classes=3)
        model.on("onTrain",  lambda: called.append("train"))
        model.on("onFinish", lambda: called.append("finish"))
        model.train(data, epochs=2, verbose=False)
        self.assertIn("train",  called)
        self.assertIn("finish", called)

    def test_invalid_event_raises(self):
        model = make_classifier()
        with self.assertRaises(OmageModelError):
            model.on("onInvalidEvent", lambda: None)


# ================================================================== #
class TestEarlyStopping(unittest.TestCase):

    def test_early_stopping_stops_training(self):
        data  = load_iris()
        model = make_classifier(num_classes=3)
        history = model.train(
            data, epochs=100, verbose=False,
            early_stopping_patience=3,
        )
        self.assertLess(len(history["loss"]), 100)


if __name__ == "__main__":
    unittest.main(verbosity=2)
