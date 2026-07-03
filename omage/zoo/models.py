"""
Omage Model Zoo — Pre-trained model loader
"""

from __future__ import annotations
import logging
import os
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger("omage")

AVAILABLE_MODELS = {
    "resnet-18":    {"type": "classifier", "backend": "torchvision", "description": "ResNet-18 — lightweight image classifier (1000 classes)", "input_size": (3, 224, 224), "num_classes": 1000},
    "resnet-50":    {"type": "classifier", "backend": "torchvision", "description": "ResNet-50 — accurate image classifier (1000 classes)",    "input_size": (3, 224, 224), "num_classes": 1000},
    "mobilenet-v3": {"type": "classifier", "backend": "torchvision", "description": "MobileNetV3-Small — fast mobile-friendly classifier",     "input_size": (3, 224, 224), "num_classes": 1000},
    "gpt2":         {"type": "generator",  "backend": "transformers", "description": "GPT-2 Small — text generation",                          "input_size": None, "num_classes": None},
    "bert-base":    {"type": "classifier", "backend": "transformers", "description": "BERT-base-uncased — text classification / embeddings",   "input_size": None, "num_classes": None},
    "distilbert":   {"type": "classifier", "backend": "transformers", "description": "DistilBERT — fast lightweight BERT (40% smaller)",       "input_size": None, "num_classes": None},
    "yolov8n":      {"type": "detector",   "backend": "ultralytics",  "description": "YOLOv8 Nano — fastest object detector (80 classes)",     "input_size": (3, 640, 640), "num_classes": 80},
    "yolov8s":      {"type": "detector",   "backend": "ultralytics",  "description": "YOLOv8 Small — balanced object detector",                "input_size": (3, 640, 640), "num_classes": 80},
}


class _ZooModel:
    """Lightweight wrapper for zoo models — no layer validation needed"""

    def __init__(self, model_type: str, torch_model: nn.Module, device):
        self.type        = model_type
        self.torch_model = torch_model.to(device)
        self.device      = device
        self.trained     = True
        self._pipeline   = None

    def predict(self, input_data):
        self.torch_model.eval()
        with torch.no_grad():
            if isinstance(input_data, torch.Tensor):
                return self.torch_model(input_data.to(self.device)).cpu().numpy()
            return f"[{self.type}] prediction"

    def evaluate(self, data):
        logger.info("Use model-specific evaluation for zoo models")
        return {"accuracy": 0.0, "loss": 0.0}

    def save(self, path: str) -> None:
        torch.save({"weights": self.torch_model.state_dict(), "type": self.type}, path)
        logger.info(f"Zoo model saved → '{path}'")

    def __repr__(self) -> str:
        return f"ZooModel(type={self.type}, trained=True)"


class ModelZoo:

    def __init__(self):
        self.models    = AVAILABLE_MODELS
        self.cache_dir = os.path.expanduser("~/.omage/models")
        os.makedirs(self.cache_dir, exist_ok=True)

    def list_models(self, filter_type: Optional[str] = None) -> None:
        print("=" * 60)
        print("  Omage Model Zoo")
        print("=" * 60)
        shown = 0
        for name, info in self.models.items():
            if filter_type and info["type"] != filter_type:
                continue
            print(f"\n  {name}")
            print(f"    Type    : {info['type']}")
            print(f"    Backend : {info['backend']}")
            print(f"    Info    : {info['description']}")
            shown += 1
        print(f"\n{'=' * 60}")
        print(f"  {shown} model(s) listed")
        print("=" * 60)

    def load(self, name: str, num_classes: Optional[int] = None, pretrained: bool = True):
        if name not in self.models:
            available = ", ".join(self.models.keys())
            raise ValueError(f"Model '{name}' not found.\nAvailable: {available}")

        info    = self.models[name]
        backend = info["backend"]
        device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Loading model: {name} (backend: {backend}, pretrained: {pretrained})")

        if backend == "torchvision":
            return self._load_torchvision(name, info, num_classes, pretrained, device)
        elif backend == "transformers":
            return self._load_transformers(name, info, device)
        elif backend == "ultralytics":
            return self._load_ultralytics(name, info, device)
        else:
            raise NotImplementedError(f"Backend '{backend}' not supported")

    def _load_torchvision(self, name, info, num_classes, pretrained, device):
        try:
            import torchvision.models as tv
        except ImportError:
            raise ImportError("torchvision not installed. Run: pip install torchvision")

        fn_map = {
            "resnet-18":    "resnet18",
            "resnet-50":    "resnet50",
            "mobilenet-v3": "mobilenet_v3_small",
        }
        fn   = getattr(tv, fn_map[name])
        base = fn(weights="DEFAULT" if pretrained else None)

        # Replace head for custom classes
        nc = num_classes or info["num_classes"]
        if nc != 1000:
            if hasattr(base, "fc"):
                base.fc = nn.Linear(base.fc.in_features, nc)
            elif hasattr(base, "classifier"):
                last = base.classifier[-1]
                base.classifier[-1] = nn.Linear(last.in_features, nc)

        logger.info(f"Loaded '{name}' from torchvision (pretrained={pretrained})")
        return _ZooModel(info["type"], base, device)

    def _load_transformers(self, name, info, device):
        try:
            from transformers import pipeline
        except ImportError:
            raise ImportError("transformers not installed. Run: pip install transformers")

        hf_map  = {"gpt2": "gpt2", "bert-base": "bert-base-uncased", "distilbert": "distilbert-base-uncased"}
        task_map = {"generator": "text-generation", "classifier": "text-classification"}
        pipe     = pipeline(task_map[info["type"]], model=hf_map[name],
                            device=0 if torch.cuda.is_available() else -1)

        m = _ZooModel(info["type"], pipe.model, device)
        m._pipeline = pipe
        logger.info(f"Loaded '{name}' from HuggingFace")
        return m

    def _load_ultralytics(self, name, info, device):
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError("ultralytics not installed. Run: pip install ultralytics")

        yolo_map = {"yolov8n": "yolov8n.pt", "yolov8s": "yolov8s.pt"}
        yolo     = YOLO(yolo_map[name])
        m        = _ZooModel("detector", yolo.model, device)
        m._yolo  = yolo
        logger.info(f"Loaded '{name}' via ultralytics")
        return m


_zoo = ModelZoo()


def load_model(name: str, num_classes: Optional[int] = None, pretrained: bool = True):
    return _zoo.load(name, num_classes, pretrained)


def list_models(filter_type: Optional[str] = None) -> None:
    _zoo.list_models(filter_type)
