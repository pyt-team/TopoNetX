"""Test Complex class."""

import pytest

from toponetx.classes.complex import Complex
from toponetx.classes.hyperedge import HyperEdge
from toponetx.classes.simplex import Simplex


class TestAtom:
    """Test the Atom class."""

    def test_atoms_equal(self):
        """Test if two atoms are equal."""
        h1 = HyperEdge((1, 2))
        s1 = Simplex((1, 2), weight=1)
        s2 = Simplex((1, 2), weight=2)
        s3 = Simplex((1, 2, 3))

        assert s1 == s1
        assert s1 == s2
        assert s1 != s3
        assert s1 != h1


class TestComplex:
    """Test the Complex abstract class."""

    def test_complex_is_abstract(self):
        """Test if the Complex abstract class is abstract."""
        with pytest.raises(TypeError):
            _ = Complex()

    def test_complex_has_abstract_methods(self):
        """Test if the Complex abstract class has all the abstract methods."""
        abstract_methods = Complex.__abstractmethods__
        abstract_methods = sorted(abstract_methods)
        assert abstract_methods == [
            "__contains__",
            "__getitem__",
            "__iter__",
            "__len__",
            "__repr__",
            "__str__",
            "add_node",
            "adjacency_matrix",
            "clone",
            "coadjacency_matrix",
            "dim",
            "incidence_matrix",
            "nodes",
            "remove_nodes",
            "shape",
            "skeleton",
        ]

    def test_complex_inheritance_check(self):
        """Test that the abstract class is forcing implementation in all children classes."""

        class ExampleClass(Complex):
            """Create an example class to test Complex Class."""

            def clone(self):
                """Test expected behavior from ExampleClass.

                Returns
                -------
                NotImplementedError
                    Currently NotImplementedError is raised.
                """
                return NotImplementedError()

        with pytest.raises(TypeError):
            _ = ExampleClass()
