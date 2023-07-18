"""Test Complex class."""

from abc import ABC, ABCMeta

import pytest

from toponetx.classes.complex import Complex


class TestComplex:
    """Test the Complex abstract class."""

    def test_complex_is_abstract(self):
        """Tests if the Complex abstract class is abstract."""
        with pytest.raises(TypeError) as exp_exception:
            Complex([1, 2])

        assert "Can't instantiate abstract class Complex with abstract methods" in str(
            exp_exception.value
        )

        assert isinstance(Complex, ABCMeta)

    def test_complex_has_abstract_methods(self):
        """Tests if the Complex abstract class has all the abstract methods."""
        abstract_methods = Complex.__abstractmethods__
        abstract_methods = sorted(list(abstract_methods))
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
            def __init__(self, name: str = "", *args, **kwargs) -> None:
                super().__init__(name, *args, **kwargs)

            def clone(self):
                return NotImplementedError

        with pytest.raises(TypeError) as exp_exception:
            ExampleClass()

        assert (
            "Can't instantiate abstract class ExampleClass with abstract methods"
            in str(exp_exception.value)
        )

    def test_complex_abstract_methods_return_none(self):
        """Test the abstract methods are returning none."""
        c = Complex
        c.__abstractmethods__ = {}
        instance_c = c()

        assert instance_c.nodes is None
        assert instance_c.dim is None
        assert instance_c.shape is None
        assert instance_c.skeleton(rank=2) is None
        assert instance_c.__str__() is None
        assert instance_c.__repr__() is None
        assert instance_c.__len__() is None
        assert instance_c.clone() is None
        assert instance_c.__iter__() is None
        assert instance_c.__contains__(item=1) is None
        assert instance_c.__getitem__(key=1) is None
        assert instance_c.remove_nodes({1, 2}) is None
        assert instance_c.add_node(1) is None
        assert instance_c.incidence_matrix() is None
        assert instance_c.adjacency_matrix() is None
        assert instance_c.coadjacency_matrix() is None
