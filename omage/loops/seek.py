"""
Omage Seek — Hyperparameter search (Grid Search)
"""

from __future__ import annotations
import copy
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger("omage")


def seek(
    model,
    data,
    epochs: int = 3,
    param_grid: Optional[Dict[str, List[Any]]] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Search for best hyperparameters using Grid Search.

    Example:
        og.seek(my_model, my_data, epochs=3, param_grid={
            "lr":         [0.1, 0.01, 0.001],
            "batch_size": [16, 32, 64],
        })
    """
    from ..core.optimizers import adam

    # Default grid if none provided
    if param_grid is None:
        param_grid = {
            "lr":         [0.1, 0.01, 0.001],
            "batch_size": [16, 32, 64],
        }

    # Build all combinations
    combinations = _build_grid(param_grid)
    total = len(combinations)

    logger.info(f"Seek started — {total} combinations | epochs/trial: {epochs}")
    if verbose:
        print(f"\n[Omage] Seek — searching {total} hyperparameter combinations...\n")

    best_loss   = float("inf")
    best_config = None
    best_metrics= None
    results     = []

    for i, config in enumerate(combinations):
        lr         = config.get("lr", 0.001)
        batch_size = config.get("batch_size", 32)

        if verbose:
            print(f"  Trial {i+1}/{total} — lr: {lr} | batch_size: {batch_size}")

        # Clone model weights so each trial starts fresh
        trial_model = _clone_model(model, lr)
        data.batch(batch_size)

        try:
            history = trial_model.train(data, epochs=epochs, verbose=False)
            metrics = trial_model.evaluate(data)
            trial_loss = metrics["loss"]
            trial_acc  = metrics["accuracy"]

            results.append({
                "config":   config,
                "loss":     trial_loss,
                "accuracy": trial_acc,
            })

            if verbose:
                print(f"    → loss: {trial_loss:.4f} | accuracy: {trial_acc:.2f}%")

            if trial_loss < best_loss:
                best_loss    = trial_loss
                best_config  = config
                best_metrics = metrics

        except Exception as exc:
            logger.warning(f"Trial {i+1} failed: {exc}")
            if verbose:
                print(f"    → failed: {exc}")
            continue

    if best_config is None:
        raise RuntimeError("All seek trials failed!")

    if verbose:
        print(f"\n[Omage] ✅ Seek complete!")
        print(f"  Best config   : {best_config}")
        print(f"  Best loss     : {best_loss:.4f}")
        print(f"  Best accuracy : {best_metrics['accuracy']:.2f}%\n")

    logger.info(f"Seek complete — best config: {best_config}")

    return {
        "best_config":  best_config,
        "best_loss":    best_loss,
        "best_metrics": best_metrics,
        "all_results":  results,
    }


def _build_grid(param_grid: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    """Build all combinations from param_grid"""
    import itertools
    keys   = list(param_grid.keys())
    values = list(param_grid.values())
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


def _clone_model(model, lr: float):
    """Create a fresh copy of the model with a new optimizer lr"""
    from ..core.model import Model
    from ..core.optimizers import adam
    import copy, torch

    new_model = Model(
        model_type=model.type,
        layers=model.layers_config,
        optimizer=adam(lr=lr),
        loss=model.loss_config,
    )
    return new_model
