"""
test_loops.py — Tests for seek, grow, polish, flow
"""

import unittest
from pathlib import Path

import omage as og

DATA_DIR = Path(__file__).parent


def make_model():
    return og.model(
        type="classifier",
        layers=[og.dense(32), og.dropout(0.2), og.dense(3, activation="softmax")],
        optimizer=og.adam(lr=0.01),
        loss=og.cross_entropy(),
    )


def load_iris():
    data = og.load(str(DATA_DIR / "iris.csv"))
    data.set_target("target").clean().normalize().shuffle(seed=42).split(0.8)
    return data


# ================================================================== #
class TestPolish(unittest.TestCase):

    def test_polish_returns_history(self):
        data  = load_iris()
        model = make_model()
        # Fresh model — train then polish separately
        history = og.polish(model, data, epochs=3, verbose=False)
        self.assertIn("loss", history)
        self.assertGreater(len(history["loss"]), 0)

    def test_polish_reduces_loss(self):
        data  = load_iris()
        model = make_model()
        # Warm up
        model.train(data, epochs=5, verbose=False)
        before = model.evaluate(data)["loss"]
        og.polish(model, data, epochs=10, verbose=False)
        after = model.evaluate(data)["loss"]
        # Loss should not drastically increase
        self.assertLess(after, before + 0.3)


# ================================================================== #
class TestSeek(unittest.TestCase):

    def test_seek_returns_best_config(self):
        data   = load_iris()
        model  = make_model()
        result = og.seek(model, data, epochs=2, param_grid={
            "lr":         [0.01, 0.001],
            "batch_size": [32],
        }, verbose=False)
        self.assertIn("best_config",  result)
        self.assertIn("best_loss",    result)
        self.assertIn("best_metrics", result)
        self.assertIn("all_results",  result)

    def test_seek_tries_all_combinations(self):
        data   = load_iris()
        model  = make_model()
        result = og.seek(model, data, epochs=2, param_grid={
            "lr":         [0.01, 0.001],
            "batch_size": [32, 64],
        }, verbose=False)
        self.assertEqual(len(result["all_results"]), 4)

    def test_seek_best_loss_is_minimum(self):
        data   = load_iris()
        model  = make_model()
        result = og.seek(model, data, epochs=2, param_grid={
            "lr": [0.1, 0.001],
            "batch_size": [32],
        }, verbose=False)
        all_losses = [r["loss"] for r in result["all_results"]]
        self.assertAlmostEqual(result["best_loss"], min(all_losses), places=4)


# ================================================================== #
class TestGrow(unittest.TestCase):

    def test_grow_returns_result(self):
        data   = load_iris()
        model  = make_model()
        result = og.grow(model, data,
                         generations=2, population=3,
                         epochs_per_gen=2, verbose=False)
        self.assertIn("generations", result)
        self.assertIn("best_score",  result)
        self.assertIn("history",     result)

    def test_grow_history_length(self):
        data   = load_iris()
        model  = make_model()
        result = og.grow(model, data,
                         generations=2, population=3,
                         epochs_per_gen=2, verbose=False)
        self.assertEqual(len(result["history"]), 2)

    def test_grow_marks_model_trained(self):
        data  = load_iris()
        model = make_model()
        og.grow(model, data, generations=2, population=3,
                epochs_per_gen=2, verbose=False)
        self.assertTrue(model.trained)

    def test_grow_best_score_positive(self):
        data   = load_iris()
        model  = make_model()
        result = og.grow(model, data, generations=2, population=3,
                         epochs_per_gen=2, verbose=False)
        self.assertGreater(result["best_score"], 0)


# ================================================================== #
class TestFlow(unittest.TestCase):

    def test_flow_returns_results(self):
        data   = load_iris()
        model1 = make_model()
        model2 = make_model()
        model1.train(data, epochs=3, verbose=False)
        model2.train(data, epochs=3, verbose=False)

        import torch
        sample  = torch.randn(4, 4)   # 4 samples, 4 features
        results = og.flow([model1, model2], sample)
        self.assertEqual(len(results), 2)

    def test_flow_single_model(self):
        data  = load_iris()
        model = make_model()
        model.train(data, epochs=3, verbose=False)
        import torch
        sample  = torch.randn(1, 4)
        results = og.flow([model], sample)
        self.assertEqual(len(results), 1)
        self.assertIsNotNone(results[0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
