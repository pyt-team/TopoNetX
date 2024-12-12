"""Test Complex class."""

import pytest

from toponetx.classes.cell import Cell
from toponetx.classes.cell_complex import CellComplex
from toponetx.classes.colored_hypergraph import ColoredHyperGraph
from toponetx.classes.complex import Atom, Complex
from toponetx.classes.hyperedge import HyperEdge
from toponetx.classes.path import Path
from toponetx.classes.path_complex import PathComplex
from toponetx.classes.simplex import Simplex
from toponetx.classes.simplicial_complex import SimplicialComplex


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

    complex_classes = (CellComplex, ColoredHyperGraph, PathComplex, SimplicialComplex)
    atom_classes = (Cell, HyperEdge, Path, Simplex)
    add_atom_method = ("add_cell", "add_cell", "add_path", "add_simplex")

    @pytest.mark.parametrize(
        "complex_class,atom_class,add_method",
        zip(complex_classes, atom_classes, add_atom_method, strict=True),
    )
    def test_add_atom_with_attribute(
        self, complex_class: type[Complex], atom_class: type[Atom], add_method: str
    ) -> None:
        """Test adding an atom with an attribute.

        Parameters
        ----------
        complex_class : type[Complex]
            The complex class to test.
        atom_class : type[Atom]
            The atom class to test.
        add_method : str
            The name of the method to add the atom to the complex.
        """
        complex_ = complex_class()
        atom1 = atom_class((1, 2, 3), weight=1)
        atom2 = atom_class((2, 3, 4))
        add_func = getattr(complex_, add_method)

        add_func(atom1)
        assert atom1 in complex_
        assert complex_[atom1]["weight"] == 1

        add_func(atom2, weight=2)
        assert atom2 in complex_
        assert complex_[atom2]["weight"] == 2

    @pytest.mark.parametrize(
        "complex_class,atom_class,add_method",
        zip(complex_classes, atom_classes, add_atom_method, strict=True),
    )
    def test_add_atom_attribute_precedence(
        self, complex_class: type[Complex], atom_class: type[Atom], add_method: str
    ) -> None:
        """Test that explicitly added attributes take precedence.

        Parameters
        ----------
        complex_class : type[Complex]
            The complex class to test.
        atom_class : type[Atom]
            The atom class to test.
        add_method : str
            The name of the method to add the atom to the complex.
        """
        complex_ = complex_class()
        atom = atom_class((1, 2, 3), weight=1)

        add_func = getattr(complex_, add_method)
        add_func(atom, weight=2)

        assert atom in complex_
        assert complex_[atom]["weight"] == 2

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
