"""
Omage Model — AI model builder and trainer
"""

from __future__ import annotations
import logging
from typing import Any, Callable, Dict, List, Optional

import torch
import torch.nn as nn

from ..utils.bridge import get_bridge, is_gpu

logger = logging.getLogger("omage")


class OmageModelError(Exception):
    pass


class Model:
    """Omage AI Model"""

    MODEL_TYPES = {
        "classifier": "classification",
        "regressor":  "regression",
        "generator":  "generation",
        "detector":   "detection",
        "predictor":  "prediction",
    }

    def __init__(
        self,
        model_type: str = "classifier",
        layers: Optional[List] = None,
        optimizer=None,
        loss=None,
        device: str = "auto",
    ):
        if model_type not in self.MODEL_TYPES:
            raise OmageModelError(
                f"Unknown model type '{model_type}'. "
                f"Available: {list(self.MODEL_TYPES.keys())}"
            )

        self.type             = model_type
        self.layers_config    = layers or []
        self.optimizer_config = optimizer
        self.loss_config      = loss
        self.bridge           = get_bridge()

        if device != "auto":
            self.bridge.set_device(device)
        self.device = self.bridge.get_device()

        self.torch_model:     Optional[nn.Module]             = None
        self.torch_optimizer: Optional[torch.optim.Optimizer] = None
        self.torch_loss:      Optional[nn.Module]             = None
        self.trained:         bool                            = False
        self.history: Dict[str, List[float]] = {
            "loss": [], "accuracy": [], "val_loss": [], "val_accuracy": []
        }
        self.handlers: Dict[str, Optional[Callable]] = {
            "onTrain": None, "onFinish": None, "onPredict": None, "onError": None,
        }
        self._build()

    def _build(self) -> None:
        if not self.layers_config:
            raise OmageModelError("No layers specified!")

        torch_layers: List[nn.Module] = []
        for layer in self.layers_config:
            if hasattr(layer, "to_torch"):
                torch_layers.append(layer.to_torch())
            elif isinstance(layer, int):
                torch_layers.extend([nn.LazyLinear(layer), nn.ReLU()])

        self.torch_model = nn.Sequential(*torch_layers)
        self.torch_model = self.bridge.move_to_device(self.torch_model)
        logger.info(f"Model built — type: {self.type} | layers: {len(self.layers_config)} | device: {self.device}")

    def compile(self, optimizer=None, loss=None) -> "Model":
        if optimizer:
            self.optimizer_config = optimizer
        if loss:
            self.loss_config = loss

        if self.optimizer_config is None:
            from ..core.optimizers import adam
            self.optimizer_config = adam()
            logger.warning("No optimizer — defaulting to Adam(lr=0.001)")

        if self.loss_config is None:
            from ..core.losses import cross_entropy
            self.loss_config = cross_entropy()
            logger.warning("No loss — defaulting to CrossEntropyLoss")

        self.torch_optimizer = self.optimizer_config.to_torch(self.torch_model.parameters())
        self.torch_loss      = self.loss_config.to_torch()
        logger.info(f"Compiled — optimizer: {self.optimizer_config} | loss: {self.loss_config}")
        return self

    def train(
        self,
        data,
        epochs: int = 10,
        batch_size: int = 32,
        verbose: bool = True,
        checkpoint_path: Optional[str] = None,
        checkpoint_every: int = 1,
        early_stopping_patience: Optional[int] = None,
    ) -> Dict[str, List[float]]:
        """Train the model with progress bars and checkpointing"""
        try:
            from tqdm import tqdm
            use_tqdm = True
        except ImportError:
            use_tqdm = False

        if self.handlers["onTrain"]:
            self.handlers["onTrain"]()

        if self.torch_optimizer is None or self.torch_loss is None:
            self.compile()

        try:
            train_loader = data.get_train_loader()
            has_val      = data.test_data is not None
            val_loader   = data.get_test_loader() if has_val else None
        except Exception as exc:
            self._trigger_error()
            raise OmageModelError(f"Failed to get data loaders: {exc}") from exc

        logger.info(f"Training — epochs: {epochs} | GPU: {is_gpu()}")

        best_val_loss      = float("inf")
        patience_counter   = 0

        try:
            epoch_iter = tqdm(range(epochs), desc="Training", unit="epoch") if use_tqdm else range(epochs)

            for epoch in epoch_iter:
                # ---- Train phase ----
                self.torch_model.train()
                epoch_loss, correct, total = 0.0, 0, 0

                batch_iter = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False) if use_tqdm else train_loader

                for batch_x, batch_y in batch_iter:
                    batch_x = batch_x.to(self.device)
                    batch_y = batch_y.to(self.device)

                    self.torch_optimizer.zero_grad()
                    outputs = self.torch_model(batch_x)
                    loss    = self.torch_loss(outputs, batch_y)
                    loss.backward()
                    self.torch_optimizer.step()

                    epoch_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total   += batch_y.size(0)
                    correct += (predicted == batch_y).sum().item()

                avg_loss = epoch_loss / len(train_loader)
                accuracy = 100.0 * correct / total if total > 0 else 0.0
                self.history["loss"].append(avg_loss)
                self.history["accuracy"].append(accuracy)

                # ---- Validation phase ----
                val_info = ""
                if val_loader:
                    v_loss, v_correct, v_total = 0.0, 0, 0
                    self.torch_model.eval()
                    with torch.no_grad():
                        for vx, vy in val_loader:
                            vx, vy = vx.to(self.device), vy.to(self.device)
                            vo      = self.torch_model(vx)
                            v_loss += self.torch_loss(vo, vy).item()
                            _, vp   = torch.max(vo.data, 1)
                            v_total   += vy.size(0)
                            v_correct += (vp == vy).sum().item()

                    v_avg   = v_loss / len(val_loader)
                    v_acc   = 100.0 * v_correct / v_total if v_total > 0 else 0.0
                    self.history["val_loss"].append(v_avg)
                    self.history["val_accuracy"].append(v_acc)
                    val_info = f" | val_loss: {v_avg:.4f} | val_acc: {v_acc:.2f}%"

                    # Early stopping
                    if early_stopping_patience:
                        if v_avg < best_val_loss:
                            best_val_loss    = v_avg
                            patience_counter = 0
                            if checkpoint_path:
                                self.save(checkpoint_path.replace(".omg", "_best.omg"))
                        else:
                            patience_counter += 1
                            if patience_counter >= early_stopping_patience:
                                logger.info(f"Early stopping at epoch {epoch+1}")
                                break

                # ---- Checkpointing ----
                if checkpoint_path and (epoch + 1) % checkpoint_every == 0:
                    ckpt = checkpoint_path.replace(".omg", f"_epoch{epoch+1}.omg")
                    self.save(ckpt)

                msg = f"Epoch {epoch+1}/{epochs} — loss: {avg_loss:.4f} | acc: {accuracy:.2f}%{val_info}"
                if use_tqdm:
                    epoch_iter.set_postfix(loss=f"{avg_loss:.4f}", acc=f"{accuracy:.2f}%")
                elif verbose:
                    logger.info(msg)

        except torch.cuda.OutOfMemoryError as exc:
            self._trigger_error()
            raise OmageModelError("GPU out of memory! Reduce batch_size or model size.") from exc
        except Exception as exc:
            self._trigger_error()
            raise OmageModelError(f"Training error: {exc}") from exc

        self.trained = True
        logger.info("Training complete!")
        if self.handlers["onFinish"]:
            self.handlers["onFinish"]()
        return self.history

    def evaluate(self, data) -> Dict[str, float]:
        if not self.trained:
            logger.warning("Model has not been trained yet!")
        test_loader = data.get_test_loader()
        self.torch_model.eval()
        correct, total, total_loss = 0, 0, 0.0
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                outputs     = self.torch_model(batch_x)
                total_loss += self.torch_loss(outputs, batch_y).item()
                _, predicted = torch.max(outputs.data, 1)
                total   += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
        accuracy = 100.0 * correct / total if total > 0 else 0.0
        avg_loss = total_loss / len(test_loader)
        metrics  = {"accuracy": accuracy, "loss": avg_loss}
        logger.info(f"Evaluation — accuracy: {accuracy:.2f}% | loss: {avg_loss:.4f}")
        return metrics

    def predict(self, input_data) -> Any:
        if self.handlers["onPredict"]:
            self.handlers["onPredict"]()
        self.torch_model.eval()
        with torch.no_grad():
            if isinstance(input_data, str):
                result = f"[{self.type}] → '{input_data}'"
            else:
                if not isinstance(input_data, torch.Tensor):
                    input_data = torch.tensor(input_data, dtype=torch.float32)
                input_data = input_data.to(self.device)
                result     = self.torch_model(input_data).cpu().numpy()
        return result

    def save(self, path: str = "model.omg") -> None:
        checkpoint = {
            "weights":          self.torch_model.state_dict(),
            "type":             self.type,
            "trained":          self.trained,
            "history":          self.history,
            "optimizer_config": str(self.optimizer_config),
            "loss_config":      str(self.loss_config),
        }
        torch.save(checkpoint, path)
        logger.info(f"Model saved → '{path}'")

    def load(self, path: str) -> "Model":
        try:
            checkpoint = torch.load(path, map_location=self.device)
        except FileNotFoundError:
            raise OmageModelError(f"Model file not found: '{path}'")
        except Exception as exc:
            raise OmageModelError(f"Failed to load '{path}': {exc}") from exc

        if isinstance(checkpoint, dict) and "weights" in checkpoint:
            self.torch_model.load_state_dict(checkpoint["weights"])
            self.trained = checkpoint.get("trained", False)
            self.history = checkpoint.get("history", self.history)
        else:
            self.torch_model.load_state_dict(checkpoint)
        logger.info(f"Model loaded ← '{path}'")
        return self

    def on(self, event: str, callback: Callable) -> "Model":
        if event not in self.handlers:
            raise OmageModelError(f"Unknown event '{event}'.")
        self.handlers[event] = callback
        return self

    def _trigger_error(self) -> None:
        if self.handlers["onError"]:
            self.handlers["onError"]()

    def __repr__(self) -> str:
        return f"Model(type={self.type}, layers={len(self.layers_config)}, trained={self.trained})"


def model(
    type: str = "classifier",
    layers: Optional[List] = None,
    optimizer=None,
    loss=None,
    device: str = "auto",
) -> Model:
    return Model(type, layers, optimizer, loss, device)
