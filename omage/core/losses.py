"""
Omage Losses — Loss functions for training
"""


class Loss:
    """Base loss class"""
    
    def __init__(self, name):
        self.name = name
    
    def to_torch(self):
        """Convert to PyTorch loss"""
        import torch.nn as nn
        
        if self.name == "cross_entropy":
            return nn.CrossEntropyLoss()
        elif self.name == "mse":
            return nn.MSELoss()
        elif self.name == "bce":
            return nn.BCELoss()
        elif self.name == "nll":
            return nn.NLLLoss()
        else:
            raise ValueError(f"Unknown loss: {self.name}")
    
    def __repr__(self):
        return self.name


def cross_entropy():
    """Cross-entropy loss (for classification)"""
    return Loss("cross_entropy")


def mse():
    """Mean squared error (for regression)"""
    return Loss("mse")


def bce():
    """Binary cross-entropy"""
    return Loss("bce")


def nll():
    """Negative log-likelihood"""
    return Loss("nll")