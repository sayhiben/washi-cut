from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List
import numpy as np

@dataclass(frozen=True)
class StripNet:
    """
    One strip's mapping from face id to 2D coordinates.
    'order' is optional; provided in Hamiltonian mode.
    """
    faces_2d: dict[int, np.ndarray]
    order: list[int] | None = None

@dataclass(frozen=True)
class UnfoldResult:
    """
    Collection of strips; each can be converted to geometry later.
    """
    strips: list[StripNet]
