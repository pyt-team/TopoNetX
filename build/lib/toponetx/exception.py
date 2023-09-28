"""Base classes for TopoNetX exceptions."""


class TopoNetXException(Exception):
    """Base class for exceptions in TopoNetX."""


class TopoNetXError(TopoNetXException):
    """Exception for a serious error in TopoNetX."""


class TopoNetXNotImplementedError(TopoNetXError):
    """Exception for methods not implemented for an object type."""
