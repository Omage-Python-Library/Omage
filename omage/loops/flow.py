"""
Omage Flow — Multi-model pipeline
"""

from __future__ import annotations
import logging
from typing import Any, List

logger = logging.getLogger("omage")


def flow(models: List, data, verbose: bool = True) -> List[Any]:
    """
    Run a pipeline of models — each model predicts independently on the input.
    Results from all models are collected and returned.

    For chained pipelines where output of model N feeds model N+1,
    use chain=True (requires compatible input/output shapes).

    Example:
        og.flow([detector, classifier, generator], my_data)
    """
    names = " → ".join(str(m) for m in models)
    if verbose:
        print(f"\n[Omage] Flow — pipeline: {names}\n")

    logger.info(f"Flow started — {len(models)} models")
    results = []

    for i, model in enumerate(models):
        if verbose:
            print(f"  Step {i+1}/{len(models)}: {model}")
        try:
            result = model.predict(data)
            results.append(result)
            if verbose:
                print(f"    → done")
        except Exception as exc:
            logger.warning(f"Flow step {i+1} failed: {exc}")
            results.append(None)
            if verbose:
                print(f"    → failed: {exc}")

    if verbose:
        print(f"\n[Omage] ✅ Flow complete — {len(results)} results\n")

    logger.info("Flow complete")
    return results
