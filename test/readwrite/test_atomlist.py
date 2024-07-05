"""Tests for the atomlist read/write functions."""

import pytest

from toponetx.classes import CellComplex, PathComplex, SimplicialComplex
from toponetx.readwrite.atomlist import (
    generate_atomlist,
    load_from_atomlist,
    parse_atomlist,
    write_atomlist,
)


class TestGenerateAtomList:
    """Test the `generate_atomlist` function."""

    def test_generate_atomlist_simplicial(self):
        """Test generate_atomlist for simplicial complexes."""
        domain = SimplicialComplex()
        domain.add_simplex((1,), weight=1.0)
        domain.add_simplex((1, 2, 3), weight=4.0)

        atomlist = set(generate_atomlist(domain))
        assert atomlist == {
            "1 {'weight': 1.0}",
            "1 2 3 {'weight': 4.0}",
        }

    def test_generate_atomlist_cell(self):
        """Test generate_atomlist for cell complexes."""
        domain = CellComplex()
        domain.add_node(1, weight=1.0)
        domain.add_node(4)
        domain.add_edge(2, 3, weight=2.0)
        domain.add_edge(2, 5)
        domain.add_cell((1, 2, 3), rank=2, weight=4.0)
        domain.add_cell((6, 7), rank=2)

        atomlist = set(generate_atomlist(domain))
        assert atomlist == {
            "1 {'weight': 1.0}",
            "4",
            "2 3 {'weight': 2.0}",
            "2 5",
            "6 7 {'rank': 2}",
            "1 2 3 {'weight': 4.0}",
        }

    def test_generate_atomlist_error(self):
        """Test generate_atomlist for erroneous inputs."""
        with pytest.raises(TypeError):
            list(generate_atomlist(PathComplex()))


class TestAtomListFileManagment:
    """Test the `write_atomlist` and `load_from_atomlist` functions."""

    def test_atomlist_simplicial(self):
        """Test that a simplicial complex can be written to and read from the filesystem as an atomlist."""
        SC = SimplicialComplex([(1, 2, 3), (2, 3, 4)])

        write_atomlist(SC, "test.atomlist")
        SC_loaded = load_from_atomlist("test.atomlist", "simplicial")
        assert isinstance(SC_loaded, SimplicialComplex)
        assert SC_loaded.shape == (4, 5, 2)

    def test_atomlist_cell(self):
        """Test that a cell complex can be written to and read from the filesystem as an atomlist."""
        CC = CellComplex()
        CC.add_cell((1, 2, 3), rank=2)
        CC.add_cell((2, 3, 4), rank=2)

        write_atomlist(CC, "test.atomlist")
        CC_loaded = load_from_atomlist("test.atomlist", "cell")
        assert isinstance(CC_loaded, CellComplex)
        assert CC_loaded.shape == (4, 5, 2)

    def test_load_from_atomlist_error(self):
        """Test that an error is raised when trying to read an atomlist with a wrong complex type."""
        SC = SimplicialComplex([(1, 2, 3), (2, 3, 4)])
        write_atomlist(SC, "test.atomlist")
        with pytest.raises(ValueError):
            load_from_atomlist("test.atomlist", "path")

    def test_write_atomlist_error(self):
        """Test that an error is raised when trying to write an atomlist with an unsupported complex type."""
        PC = PathComplex([(1, 2)])
        with pytest.raises(TypeError):
            write_atomlist(PC, "test.atomlist")


class TestParseAtomList:
    """Test the `parse_atomlist` function."""

    def test_parse_atomlist_simplicial(self):
        """Test parse_atomlist for simplicial complexes."""
        # empty atomlist
        SC = parse_atomlist([], "simplicial")
        assert isinstance(SC, SimplicialComplex)
        assert SC.shape == ()

        # atomlist with one simplex
        SC = parse_atomlist(["1 2 3 {'weight': 4.0}"], "simplicial")
        assert isinstance(SC, SimplicialComplex)
        assert SC.shape == (3, 3, 1)
        assert SC[("1", "2", "3")]["weight"] == 4.0

        # nodetype
        SC = parse_atomlist(["1 2 3 {'weight': 4.0}"], "simplicial", nodetype=int)
        assert SC[(1, 2, 3)]["weight"] == 4.0

    def test_parse_atomlist_cell(self):
        """Test parse_atomlist for cell complexes."""
        # empty atomlist
        CC = parse_atomlist([], "cell")
        assert isinstance(CC, CellComplex)
        assert CC.shape == (0, 0, 0)

        # atomlist with one cell
        CC = parse_atomlist(["1 2 3 {'weight': 4.0}"], "cell")
        assert isinstance(CC, CellComplex)
        assert CC.shape == (3, 3, 1)
        assert CC.cells[("1", "2", "3")]["weight"] == 4.0

        # one node
        CC = parse_atomlist(["1 {'weight': 4.0}"], "cell")
        assert CC.shape == (1, 0, 0)
        assert CC.nodes["1"]["weight"] == 4.0

        # 2-element cell with rank 2
        CC = parse_atomlist(["1 2 {'rank': 2}"], "cell")
        assert CC.shape == (2, 1, 1)

        # nodetype
        CC = parse_atomlist(["1 2 3 {'weight': 4.0}"], "cell", nodetype=int)
        assert CC.cells[(1, 2, 3)]["weight"] == 4.0

    def test_parse_atomlist_error(self):
        """Test parse_atomlist for erroneous inputs."""
        with pytest.raises(ValueError):
            parse_atomlist([], "path")
