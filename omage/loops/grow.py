"""
Omage Grow — Evolutionary training (Genetic Algorithm)
"""

from __future__ import annotations
import copy
import logging
import random
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

logger = logging.getLogger("omage")


def grow(
    model,
    data,
    generations:    int   = 10,
    survival_rate:  float = 0.2,
    population:     int   = 6,
    mutation_rate:  float = 0.1,
    epochs_per_gen: int   = 3,
    verbose:        bool  = True,
) -> Dict[str, Any]:
    """
    Train using a Genetic Algorithm — evolves a population of models.

    Each generation:
      1. Train all models briefly (initializes lazy layers)
      2. Keep top survival_rate% (elites)
      3. Fill the rest with mutated copies of elites
      4. Repeat

    Example:
        og.grow(my_model, my_data, generations=10, survival_rate=0.2)
    """
    logger.info(f"Grow started — generations: {generations} | population: {population}")
    if verbose:
        print(f"\n[Omage] Grow — Evolutionary Training")
        print(f"  Generations   : {generations}")
        print(f"  Population    : {population}")
        print(f"  Survival rate : {survival_rate*100:.0f}%")
        print(f"  Mutation rate : {mutation_rate*100:.0f}%\n")

    # Build initial population — each trains first to initialize lazy layers
    pop = [_clone_fresh(model) for _ in range(population)]

    n_survivors = max(1, int(population * survival_rate))
    history     = []
    best_weights = None
    best_score   = float("inf")

    for gen in range(generations):
        if verbose:
            print(f"  Generation {gen+1}/{generations}")

        scores = []
        for i, member in enumerate(pop):
            try:
                # Train also initializes LazyLinear layers
                member.train(data, epochs=epochs_per_gen, verbose=False)
                metrics = member.evaluate(data)
                scores.append((metrics["loss"], metrics["accuracy"], i))
            except Exception as exc:
                logger.warning(f"Gen {gen+1} member {i} failed: {exc}")
                scores.append((float("inf"), 0.0, i))

        scores.sort(key=lambda x: x[0])
        gen_best_loss = scores[0][0]
        gen_best_acc  = scores[0][1]
        history.append({"generation": gen + 1, "loss": gen_best_loss, "accuracy": gen_best_acc})

        if verbose:
            print(f"    Best loss: {gen_best_loss:.4f} | Best acc: {gen_best_acc:.2f}%")

        if gen_best_loss < best_score:
            best_score   = gen_best_loss
            best_weights = copy.deepcopy(pop[scores[0][2]].torch_model.state_dict())

        # Elitism
        elite_indices = [s[2] for s in scores[:n_survivors]]
        elites = [pop[i] for i in elite_indices]

        # Next generation
        next_pop = list(elites)
        while len(next_pop) < population:
            parent = random.choice(elites)
            child  = _clone_trained(parent)   # clone already-initialized model
            _mutate(child, mutation_rate)
            next_pop.append(child)

        pop = next_pop

    if verbose:
        print(f"\n[Omage] ✅ Grow complete!")
        print(f"  Best loss across {generations} generations: {best_score:.4f}\n")

    logger.info(f"Grow complete — best loss: {best_score:.4f}")

    # Copy best weights back to original model
    if best_weights is not None:
        # Initialize original model first if needed
        try:
            train_loader = data.get_train_loader()
            for bx, _ in train_loader:
                model.torch_model(bx.to(model.device))
                break
        except Exception:
            pass
        try:
            model.torch_model.load_state_dict(best_weights)
        except Exception as exc:
            logger.warning(f"Could not copy best weights back: {exc}")
        model.trained = True

    return {
        "generations": generations,
        "best_score":  best_score,
        "history":     history,
    }


def _clone_fresh(model):
    """Create a new untrained model with same architecture"""
    from ..core.model import Model
    from ..core.optimizers import adam
    from ..core.losses import cross_entropy

    return Model(
        model_type=model.type,
        layers=model.layers_config,
        optimizer=model.optimizer_config or adam(),
        loss=model.loss_config or cross_entropy(),
    )


def _clone_trained(model):
    """Clone an already-trained (initialized) model with its weights"""
    from ..core.model import Model
    import copy

    new_model = Model(
        model_type=model.type,
        layers=model.layers_config,
        optimizer=model.optimizer_config,
        loss=model.loss_config,
    )
    # Initialize by copying state_dict (works because parent is already initialized)
    try:
        new_model.torch_model.load_state_dict(
            copy.deepcopy(model.torch_model.state_dict())
        )
    except Exception:
        pass  # will be initialized on first train()
    return new_model


def _mutate(model, mutation_rate: float) -> None:
    """Apply small random noise to initialized weights"""
    try:
        with torch.no_grad():
            for param in model.torch_model.parameters():
                if not isinstance(param, torch.nn.parameter.UninitializedParameter):
                    if random.random() < mutation_rate:
                        noise = torch.randn_like(param) * 0.05
                        param.add_(noise)
    except Exception as exc:
        logger.debug(f"Mutation skipped: {exc}")
