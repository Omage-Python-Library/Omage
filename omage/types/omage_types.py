"""
Omage Types — Type definitions
"""

from enum import Enum


class ModelType(Enum):
    """Supported model types"""
    CLASSIFIER = "classifier"
    REGRESSOR = "regressor"
    GENERATOR = "generator"
    DETECTOR = "detector"
    PREDICTOR = "predictor"


class OmageType:
    """Base type for Omage values"""
    
    def __init__(self, value, type_name):
        self.value = value
        self.type = type_name
    
    def __repr__(self):
        return f"{self.type}({self.value})"


class Tensor(OmageType):
    """Tensor type"""
    
    def __init__(self, data):
        import numpy as np
        import torch
        
        if isinstance(data, list):
            data = np.array(data)
        
        if isinstance(data, np.ndarray):
            data = torch.tensor(data, dtype=torch.float32)
        
        super().__init__(data, "tensor")
        self.tensor = data
    
    def shape(self):
        return self.tensor.shape
    
    def to_numpy(self):
        return self.tensor.numpy()


class Layer(OmageType):
    """Layer type"""
    
    def __init__(self, layer_obj):
        super().__init__(layer_obj, "layer")
        self.layer = layer_obj