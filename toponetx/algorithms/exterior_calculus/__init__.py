"""
Exterior calculus algorithms for TopoNetX.

This subpackage exposes:
- `ExteriorCalculusOperators`: DEC coboundaries, Hodge stars, codifferentials, Laplacians.
- `MetricSpec`: user-facing specification for diagonal stars and anisotropic tensors.
"""

from __future__ import annotations

from .exterior_calc import ExteriorCalculusOperators
from .metric import MetricSpec

__all__ = ["ExteriorCalculusOperators", "MetricSpec"]
