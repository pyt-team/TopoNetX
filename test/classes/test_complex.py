"""Test Complex class."""

from abc import ABCMeta

import pytest

from toponetx.classes.complex import Complex


class TestComplex:
    """Test the Complex abstract class."""

    def test_complex_is_abstract(self):
        """Test if the Complex abstract class is abstract."""
        with pytest.raises(TypeError) as exp_exception:
            Complex()

        assert "Can't instantiate abstract class Complex with abstract methods" in str(
            exp_exception.value
        )

        assert isinstance(Complex, ABCMeta)

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
            """Create an example class to test Complex Class.

            Parameters
            ----------
                name : str
                    Name of the Complex.
                *args
                    Positional arguments to be passed to the constructor.
                **kwargs
                    Keyword arguments to be passed to the constructor.
            """

            def __init__(self, name: str = "", *args, **kwargs) -> None:
                """Initialize the example class.

                Parameters
                ----------
                name : str
                    Name of the Complex.
                *args
                    Positional arguments to be passed to the constructor.
                **kwargs
                    Keyword arguments to be passed to the constructor.

                Returns
                -------
                ExampleClass
                    The example class is initialized and returned.
                """
                super().__init__(name, *args, **kwargs)

            def clone(self):
                """Test expected behavior from ExampleClass.

                Returns
                -------
                NotImplementedError
                    Currently NotImplementedError is raised.
                """
                return NotImplementedError

        with pytest.raises(TypeError) as exp_exception:
            ExampleClass()

        assert (
            "Can't instantiate abstract class ExampleClass with abstract methods"
            in str(exp_exception.value)
        )
