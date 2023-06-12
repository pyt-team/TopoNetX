"""Test Complex class."""

import unittest
from abc import ABC, abstractmethod

from toponetx.classes.complex import Complex


class TestComplex(unittest.TestCase):
    """Test the Complex abstract class."""

    class ConcreteComplex(Complex):
        """Concrete implementation of the Complex abstract class for tests."""

        @property
        def nodes(self):
            """Nodes."""
            raise NotImplementedError

        @property
        def dim(self) -> int:
            """Dimension."""
            raise NotImplementedError

        def shape(self):
            """Compute shape."""
            raise NotImplementedError

        def skeleton(self, rank):
            """Compute rank-skeleton."""
            raise NotImplementedError

        def __str__(self):
            """Compute string representation."""
            raise NotImplementedError

        def __repr__(self):
            """Compute string representation."""
            raise NotImplementedError

        def __len__(self) -> int:
            """Compute number of nodes."""
            raise NotImplementedError

        def clone(self):
            """Clone the complex."""
            raise NotImplementedError

        def __iter__(self):
            """Iterate over the nodes."""
            raise NotImplementedError

        def __contains__(self, item):
            """Check if a node is in the complex."""
            raise NotImplementedError

        def __getitem__(self, node):
            """Get the node object from the complex."""
            raise NotImplementedError

        def remove_nodes(self, node_set):
            """Remove nodes from the complex."""
            raise NotImplementedError

        def add_node(self, node):
            """Add a node to the complex."""
            raise NotImplementedError

        def incidence_matrix(self):
            """Compute the incidence matrix."""
            raise NotImplementedError

        def adjacency_matrix(self):
            """Compute the adjacency matrix."""
            raise NotImplementedError

        def coadjacency_matrix(self):
            """Compute the coadjacency matrix."""
            raise NotImplementedError

    def test_abstract_methods(self):
        """Test that abstract methods raise NotImplementedError."""
        complex_obj = self.ConcreteComplex()

        with self.assertRaises(NotImplementedError):
            complex_obj.nodes

        with self.assertRaises(NotImplementedError):
            complex_obj.dim

        with self.assertRaises(NotImplementedError):
            complex_obj.shape()

        with self.assertRaises(NotImplementedError):
            complex_obj.skeleton(0)

        with self.assertRaises(NotImplementedError):
            str(complex_obj)

        with self.assertRaises(NotImplementedError):
            repr(complex_obj)

        with self.assertRaises(NotImplementedError):
            len(complex_obj)

        with self.assertRaises(NotImplementedError):
            complex_obj.clone()

        with self.assertRaises(NotImplementedError):
            iter(complex_obj)

        with self.assertRaises(NotImplementedError):
            1 in complex_obj

        with self.assertRaises(NotImplementedError):
            complex_obj["node"]

        with self.assertRaises(NotImplementedError):
            complex_obj.remove_nodes([1, 2, 3])

        with self.assertRaises(NotImplementedError):
            complex_obj.add_node("node")

        with self.assertRaises(NotImplementedError):
            complex_obj.incidence_matrix()

        with self.assertRaises(NotImplementedError):
            complex_obj.adjacency_matrix()

        with self.assertRaises(NotImplementedError):
            complex_obj.coadjacency_matrix()


if __name__ == "__main__":
    unittest.main()
