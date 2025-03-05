"""
PNAS (Progressive Neural Architecture Search) Implementation

This module implements the PNAS algorithm as described in the paper:
"Progressive Neural Architecture Search" by Liu et al. (2018)
https://arxiv.org/abs/1712.00559

It includes a surrogate model for performance prediction and progressive 
widening strategy for efficient architecture search.
"""

from .surrogate_model import SurrogateModel
from .pnas_search import PNASSearch