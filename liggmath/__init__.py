"""liggmath: linear algebra and AI math utilities.

Submodules:
    liggmath.linalg  - vectors, matrices, decompositions, linear systems
    liggmath.stats   - descriptive statistics and probability distributions
    liggmath.ai      - activations, losses, metrics, autograd, optimizers, init
    liggmath.utils   - numerical helpers (gradient checking, data utils)
"""

from . import linalg, stats, ai, utils

__version__ = "0.1.0"

__all__ = ["linalg", "stats", "ai", "utils", "__version__"]
