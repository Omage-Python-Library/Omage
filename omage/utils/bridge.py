"""
Omage Bridge — GPU and backend management
"""

import logging
import torch

logger = logging.getLogger("omage")


class Bridge:
    """Bridge between Omage and PyTorch backend"""

    def __init__(self, backend: str = "pytorch"):
        self.backend = backend
        self.device = self._auto_detect_device()

    def _auto_detect_device(self) -> torch.device:
        """Automatically detect best available device"""
        if torch.cuda.is_available():
            logger.info(f"GPU detected: {torch.cuda.get_device_name(0)}")
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            logger.info("Apple Silicon GPU detected")
            return torch.device("mps")
        else:
            logger.info("No GPU found, using CPU")
            return torch.device("cpu")

    def get_device(self) -> torch.device:
        return self.device

    def set_device(self, device_name: str) -> torch.device:
        if device_name == "cuda" and torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif device_name == "mps" and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif device_name == "cpu":
            self.device = torch.device("cpu")
        else:
            logger.warning(f"Device '{device_name}' not available, keeping {self.device}")
        logger.info(f"Device set to: {self.device}")
        return self.device

    def move_to_device(self, obj):
        return obj.to(self.device)

    def is_gpu(self) -> bool:
        return self.device.type in ("cuda", "mps")

    def get_backend(self) -> str:
        return self.backend

    def set_backend(self, backend: str) -> None:
        if backend not in ("pytorch",):
            raise ValueError(f"Unsupported backend: '{backend}'. Available: 'pytorch'")
        self.backend = backend
        logger.info(f"Backend switched to: {backend}")


# Singleton
_default_bridge = Bridge("pytorch")


def get_bridge() -> Bridge:
    return _default_bridge

def get_device() -> torch.device:
    return _default_bridge.get_device()

def set_device(device_name: str) -> torch.device:
    return _default_bridge.set_device(device_name)

def is_gpu() -> bool:
    return _default_bridge.is_gpu()
