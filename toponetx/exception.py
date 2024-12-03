"""Base errors and exceptions for TopoNetX."""

__all__ = [
    "TopoNetXAlgorithmError",
    "TopoNetXException",
    "TopoNetXNoPath",
    "TopoNetXUnfeasible",
]


class TopoNetXException(Exception):
    """Base class for exceptions in TopoNetX."""


class TopoNetXAlgorithmError(TopoNetXException):
    """Exception that is raised if an algorithm terminates unexpectedly."""


class TopoNetXUnfeasible(TopoNetXAlgorithmError):
    """Exception raised by algorithms trying to solve a problem instance that has no feasible solution."""


class TopoNetXNoPath(TopoNetXUnfeasible):
    """Exception for algorithms that should return a path or path length where such a path does not exist."""
