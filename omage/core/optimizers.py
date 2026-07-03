"""
Omage Optimizers — Training optimizers
"""


class Optimizer:
    """Base optimizer class"""
    
    def __init__(self, name, **kwargs):
        self.name = name
        self.config = kwargs
    
    def to_torch(self, model_parameters):
        """Convert to PyTorch optimizer"""
        import torch.optim as optim
        
        if self.name == "adam":
            return optim.Adam(model_parameters, **self.config)
        elif self.name == "sgd":
            return optim.SGD(model_parameters, **self.config)
        elif self.name == "rmsprop":
            return optim.RMSprop(model_parameters, **self.config)
        else:
            raise ValueError(f"Unknown optimizer: {self.name}")
    
    def __repr__(self):
        params = ", ".join(f"{k}={v}" for k, v in self.config.items())
        return f"{self.name}({params})"


def adam(lr=0.001, beta1=0.9, beta2=0.999, weight_decay=0):
    """Adam optimizer"""
    return Optimizer("adam", lr=lr, betas=(beta1, beta2), weight_decay=weight_decay)


def sgd(lr=0.01, momentum=0, weight_decay=0):
    """SGD optimizer"""
    return Optimizer("sgd", lr=lr, momentum=momentum, weight_decay=weight_decay)


def rmsprop(lr=0.001, alpha=0.99, weight_decay=0):
    """RMSprop optimizer"""
    return Optimizer("rmsprop", lr=lr, alpha=alpha, weight_decay=weight_decay)