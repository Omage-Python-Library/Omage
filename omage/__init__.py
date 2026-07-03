"""
Omage — AI-First Python Library
Build AI the way you think.
"""

import logging

__version__ = "0.1.1"

# Setup library-level logger (silent by default — user can enable)
logging.getLogger("omage").addHandler(logging.NullHandler())

# Core
from .core.data       import Data, load
from .core.model      import Model, model
from .core.layers     import dense, dropout, conv2d, flatten, pool
from .core.optimizers import adam, sgd, rmsprop
from .core.losses     import cross_entropy, mse, bce, nll
from .core.train      import train

# AI Loops
from .loops.polish import polish
from .loops.seek   import seek
from .loops.flow   import flow
from .loops.grow   import grow

# Utils
from .utils.bridge import get_device, set_device, is_gpu

# Model Zoo
from .zoo.models import load_model, list_models

# Compiler
from .compiler.transpiler import transpile, compile_file

__all__ = [
    "__version__",
    # data
    "Data", "load",
    # model
    "Model", "model", "train",
    # layers
    "dense", "dropout", "conv2d", "flatten", "pool",
    # optimizers
    "adam", "sgd", "rmsprop",
    # losses
    "cross_entropy", "mse", "bce", "nll",
    # loops
    "polish", "seek", "flow", "grow",
    # utils
    "get_device", "set_device", "is_gpu",
    # zoo
    "load_model", "list_models",
    # compiler
    "transpile", "compile_file",
]
