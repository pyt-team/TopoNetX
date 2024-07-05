"""Configure our testing suite."""

import networkx
import numpy
import pytest

import toponetx


@pytest.fixture(autouse=True)
def doctest_default_imports(doctest_namespace):
    """Add default imports to the doctest namespace.

    This fixture adds the following default imports to every doctest, so that their use
    is consistent across all doctests without boilerplate imports polluting the
    doctests themselves:

    .. code-block:: python

        import numpy as np
        import networkx as nx
        import toponetx as tnx

    Parameters
    ----------
    doctest_namespace : dict
        The namespace of the doctest.
    """
    doctest_namespace["np"] = numpy
    doctest_namespace["nx"] = networkx
    doctest_namespace["tnx"] = toponetx
