"""Configure our testing suite."""

import networkx as nx
import numpy as np
import pytest

import toponetx as tnx


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
    doctest_namespace["np"] = np
    doctest_namespace["nx"] = nx
    doctest_namespace["tnx"] = tnx
