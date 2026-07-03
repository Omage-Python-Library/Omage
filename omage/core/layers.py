"""
Omage Layers — Building blocks for AI models
"""

import torch
import torch.nn as nn


class Layer:
    """Base class for all layers"""
    
    def __init__(self, name, **kwargs):
        self.name = name
        self.config = kwargs
    
    def to_torch(self):
        """Convert to PyTorch layer"""
        raise NotImplementedError("Subclasses must implement to_torch()")
    
    def __repr__(self):
        params = ", ".join(f"{k}={v}" for k, v in self.config.items())
        return f"{self.name}({params})"


class DenseLayer(Layer):
    """Fully connected layer (Linear + Activation)"""
    
    def __init__(self, units, activation="relu", input_dim=None):
        super().__init__("dense", units=units, activation=activation)
        self.units = units
        self.activation = activation
        self.input_dim = input_dim
    
    def to_torch(self):
        layers = []
        if self.input_dim:
            layers.append(nn.Linear(self.input_dim, self.units))
        else:
            layers.append(nn.LazyLinear(self.units))
        
        if self.activation == "relu":
            layers.append(nn.ReLU())
        elif self.activation == "sigmoid":
            layers.append(nn.Sigmoid())
        elif self.activation == "tanh":
            layers.append(nn.Tanh())
        elif self.activation == "softmax":
            layers.append(nn.Softmax(dim=1))
        elif self.activation is None or self.activation == "none":
            pass  # No activation
        
        return nn.Sequential(*layers)


class DropoutLayer(Layer):
    """Dropout layer for regularization"""
    
    def __init__(self, rate=0.2):
        super().__init__("dropout", rate=rate)
        self.rate = rate
    
    def to_torch(self):
        return nn.Dropout(p=self.rate)


class Conv2DLayer(Layer):
    """2D Convolutional layer"""
    
    def __init__(self, filters, kernel_size=3, stride=1, padding=0, activation="relu"):
        super().__init__("conv2d", filters=filters, kernel_size=kernel_size)
        self.filters = filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.activation = activation
    
    def to_torch(self):
        # Note: in_channels will be determined at runtime
        return nn.LazyConv2d(
            out_channels=self.filters,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding
        )


class FlattenLayer(Layer):
    """Flatten layer"""
    
    def __init__(self):
        super().__init__("flatten")
    
    def to_torch(self):
        return nn.Flatten()


class PoolLayer(Layer):
    """Pooling layer"""
    
    def __init__(self, pool_size=2, pool_type="max"):
        super().__init__("pool", pool_size=pool_size, pool_type=pool_type)
        self.pool_size = pool_size
        self.pool_type = pool_type
    
    def to_torch(self):
        if self.pool_type == "max":
            return nn.MaxPool2d(kernel_size=self.pool_size)
        else:
            return nn.AvgPool2d(kernel_size=self.pool_size)


# Helper functions — user-friendly API
def dense(units, activation="relu", input_dim=None):
    """Create a dense layer"""
    return DenseLayer(units, activation, input_dim)


def dropout(rate=0.2):
    """Create a dropout layer"""
    return DropoutLayer(rate)


def conv2d(filters, kernel_size=3, stride=1, padding=0, activation="relu"):
    """Create a 2D convolutional layer"""
    return Conv2DLayer(filters, kernel_size, stride, padding, activation)


def flatten():
    """Create a flatten layer"""
    return FlattenLayer()


def pool(pool_size=2, pool_type="max"):
    """Create a pooling layer"""
    return PoolLayer(pool_size, pool_type)