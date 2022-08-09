# Copyright © 2022 Pyt-Team
# All rights reserved.

"""
Base classes for TopoNetX exceptions
"""


class TopoNetXException(Exception):
    """Base class for exceptions in HyperNetX."""


class TopoNetXError(TopoNetXException):
    """Exception for a serious error in HyperNetX"""


class TopoNetXNotImplementedError(TopoNetXError):
    """Exception for methods not implemented for an object type."""
