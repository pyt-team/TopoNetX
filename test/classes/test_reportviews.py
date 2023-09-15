"""Test ReportViews module."""

import pytest

from toponetx.classes import Cell, CellComplex, CombinatorialComplex, HyperEdge
from toponetx.classes.reportviews import CellView, HyperEdgeView, NodeView, SimplexView
from toponetx.exception import TopoNetXError


class TestReportViews_CellView:
    """Test Cell View Class of the ReportViews module."""

    CX = CellComplex()
    CX._insert_cell((1, 2, 3, 4), color="red")
    CX._insert_cell((1, 2, 3, 4), color="green")
    CX._insert_cell((1, 2, 3, 4), shape="square")
    CX._insert_cell((2, 3, 4, 5))

    CV = CX._cells
    cell_1 = Cell((1, 2, 3, 4))
    cell_2 = Cell((1, 2, 3, 8))

    def test_cell_view_report_getitem_method(self):
        """Test the __getitem__ method of CellView class."""
        with pytest.raises(KeyError) as exp_exception:
            self.CV.__getitem__(self.cell_2)

        assert (
            str(exp_exception.value)
            == "'cell Cell((1, 2, 3, 8)) is not in the cell dictionary'"
        )

        assert list(self.CV.__getitem__(self.cell_1)) == [
            {"color": "red"},
            {"color": "green"},
            {"shape": "square"},
        ]

        with pytest.raises(KeyError) as exp_exception:
            self.CV.__getitem__((1, 2, 3, 8))

        assert (
            str(exp_exception.value)
            == "'cell (1, 2, 3, 8) is not in the cell dictionary'"
        )

        with pytest.raises(TypeError) as exp_exception:
            self.CV.__getitem__(1)

        assert str(exp_exception.value) == "Input must be a tuple, list or a cell."

    def test_cell_view_raw_method_1(self):
        """Test the raw method for the CellView class."""
        with pytest.raises(TypeError) as exp_exception:
            self.CV.raw(1)

        assert str(exp_exception.value) == "Input must be a tuple, list or a cell."

    def test_cell_view_raw_method_2(self):
        """Test the raw method for the CellView class."""
        # Raw Method should return an exception when argument type is int
        with pytest.raises(TypeError) as exp_exception:
            self.CV.raw(2)

        assert str(exp_exception.value) == "Input must be a tuple, list or a cell."

    def test_cell_view_raw_method_3(self):
        """Test the raw method for the CellView class."""
        # Raw Method should return an exception when argument type is string
        with pytest.raises(TypeError) as exp_exception:
            self.CV.raw("testing string")

        assert str(exp_exception.value) == "Input must be a tuple, list or a cell."

    def test_cell_view_raw_method_4(self):
        """Test the raw method for the CellView class."""
        # Raw Method should return an exception when argument type is set
        with pytest.raises(TypeError) as exp_exception:
            self.CV.raw({1})

        assert str(exp_exception.value) == "Input must be a tuple, list or a cell."

    def test_cell_view_raw_method_5(self):
        """Test the raw method for the CellView class."""
        # Raw Method should return an exception when argument type is dict
        with pytest.raises(TypeError) as exp_exception:
            self.CV.raw({"1": "some_string"})

        assert str(exp_exception.value) == "Input must be a tuple, list or a cell."

    def test_cell_view_raw_method_6(self):
        """Test the raw method for the CellView class."""
        # Raw Method should raise a KeyError exception when cell is not present
        with pytest.raises(KeyError) as exp_exception:
            self.CV.raw(self.cell_2)

        assert (
            str(exp_exception.value)
            == "'cell Cell((1, 2, 3, 8)) is not in the cell dictionary'"
        )

    def test_cell_view_raw_method_7(self):
        """Test the raw method for the CellView class."""
        assert str(self.CV.raw(self.cell_1)) == str(
            [Cell((1, 2, 3, 4)), Cell((1, 2, 3, 4)), Cell((1, 2, 3, 4))]
        )

    def test_cell_view_raw_method_8(self):
        """Test the raw method for the CellView class."""
        with pytest.raises(KeyError) as exp_exception:
            self.CV.raw((1, 2, 3, 8))

        assert (
            str(exp_exception.value)
            == "'cell (1, 2, 3, 8) is not in the cell dictionary'"
        )

    def test_cell_view_contains_method(self):
        """Test the contains method for the CellView class."""
        CX = CellComplex()
        # Inserting some cells in CX
        CX._insert_cell((1, 2, 3, 4), color="red")
        CX._insert_cell((1, 2, 3, 4), color="green")
        CX._insert_cell((7, 8, 9, 1), shape="square")
        CX._insert_cell((2, 3, 4, 5))

        # Creating Cell View from Cell Complex
        CV = CX._cells

        # To test we are using three cells that are present in the cell complex
        # these are the three cases of multiple homotopic cells present (with properties)
        # and single cell (with and without properties attribute)

        cell_1 = Cell((1, 2, 3, 4))
        cell_2 = Cell((2, 3, 4, 1))
        cell_3 = Cell((1, 3, 2, 4))
        cell_4 = Cell((7, 8, 9, 1))
        cell_5 = Cell((1, 7, 8, 9))
        cell_6 = Cell((7, 1, 8, 9))
        cell_7 = Cell((2, 3, 4, 5))
        cell_8 = Cell((3, 4, 5, 2))
        cell_9 = Cell((3, 2, 4, 5))
        cell_10 = Cell((1, 2, 3, 8))

        assert CV.__contains__(cell_1) is True
        assert CV.__contains__(cell_2) is True
        assert CV.__contains__(cell_3) is False

        assert CV.__contains__(cell_4) is True
        assert CV.__contains__(cell_5) is True
        assert CV.__contains__(cell_6) is False

        assert CV.__contains__(cell_7) is True
        assert CV.__contains__(cell_8) is True
        assert CV.__contains__(cell_9) is False

        assert CV.__contains__(cell_10) is False

        # Now testing the same with list and tuples as inputs

        cell_1_tp = (1, 2, 3, 4)
        cell_2_tp = (2, 3, 4, 1)
        cell_3_tp = (1, 3, 2, 4)
        cell_4_tp = (7, 8, 9, 1)
        cell_5_tp = (1, 7, 8, 9)
        cell_6_tp = (7, 1, 8, 9)
        cell_7_tp = (2, 3, 4, 5)
        cell_8_tp = (3, 4, 5, 2)
        cell_9_tp = (3, 2, 4, 5)
        cell_10_tp = (1, 2, 3, 8)

        assert CV.__contains__(cell_1_tp) is True
        assert CV.__contains__(cell_2_tp) is True
        assert CV.__contains__(cell_3_tp) is False

        assert CV.__contains__(cell_4_tp) is True
        assert CV.__contains__(cell_5_tp) is True
        assert CV.__contains__(cell_6_tp) is False

        assert CV.__contains__(cell_7_tp) is True
        assert CV.__contains__(cell_8_tp) is True
        assert CV.__contains__(cell_9_tp) is False

        assert CV.__contains__(cell_10_tp) is False

        assert CV.__contains__(list(cell_1_tp)) is True
        assert CV.__contains__(list(cell_2_tp)) is True
        assert CV.__contains__(list(cell_3_tp)) is False

        assert CV.__contains__(list(cell_4_tp)) is True
        assert CV.__contains__(list(cell_5_tp)) is True
        assert CV.__contains__(list(cell_6_tp)) is False

        assert CV.__contains__(list(cell_7_tp)) is True
        assert CV.__contains__(list(cell_8_tp)) is True
        assert CV.__contains__(list(cell_9_tp)) is False

        assert CV.__contains__(list(cell_10_tp)) is False

        with pytest.raises(TypeError) as exp_exception:
            CV.__contains__({1, 2, 3, 4})

        assert (
            str(exp_exception.value) == "Input must be of type: tuple, list or a cell."
        )

        with pytest.raises(TypeError) as exp_exception:
            CV.__contains__(1)

        assert (
            str(exp_exception.value) == "Input must be of type: tuple, list or a cell."
        )

        with pytest.raises(TypeError) as exp_exception:
            CV.__contains__("1234")

        assert (
            str(exp_exception.value) == "Input must be of type: tuple, list or a cell."
        )

        with pytest.raises(TypeError) as exp_exception:
            CV.__contains__({"1": True, "2": True, "3": True, "4": True})

        assert (
            str(exp_exception.value) == "Input must be of type: tuple, list or a cell."
        )

    def test_cell_view_repr_method(self):
        """Test the __repr__ method for the CellView class."""
        assert (
            self.CV.__repr__()
            == "CellView([Cell((1, 2, 3, 4)), Cell((1, 2, 3, 4)), Cell((1, 2, 3, 4)), Cell((2, 3, 4, 5))])"
        )

    def test_cell_view_str_method(self):
        """Test the __str__ method for the CellView class."""
        assert (
            self.CV.__str__()
            == "CellView([Cell((1, 2, 3, 4)), Cell((1, 2, 3, 4)), Cell((1, 2, 3, 4)), Cell((2, 3, 4, 5))])"
        )


class TestReportViews_HyperEdgeView:
    """Test the HyperEdgeView class of the ReportView class."""

    def test_hyper_edge_view_to_frozenset(self):
        """Testing the _to_frozen_set static method of the Hyperedge View Class."""
        he_1 = HyperEdge((1, 2, 3, 4), rank=2)
        he_2 = HyperEdge({1, 2, 3, 4}, rank=2)
        he_3 = [1, 2, 3, 4]
        he_4 = (1, 2, 3, 4)
        he_5 = 1

        assert HyperEdgeView._to_frozen_set(he_1) == frozenset({1, 2, 3, 4})
        assert HyperEdgeView._to_frozen_set(he_2) == frozenset({1, 2, 3, 4})
        assert HyperEdgeView._to_frozen_set(he_3) == frozenset({1, 2, 3, 4})
        assert HyperEdgeView._to_frozen_set(he_4) == frozenset({1, 2, 3, 4})
        assert HyperEdgeView._to_frozen_set(he_5) == frozenset({1})

    def test_hyper_edge_view_contains(self):
        """Test the __contains__ method for the Hyperedge View Class."""
        hev = HyperEdgeView(name="testing_hev")

        he_1 = HyperEdge((1, 2, 3, 4), rank=2)
        he_2 = HyperEdge({1, 2, 3, 5}, rank=2)
        he_3 = [1, 2, 3, 4]
        he_4 = (1, 2, 3, 5)
        he_5 = []
        he_6 = HyperEdge({}, rank=1)

        CC = CombinatorialComplex([he_1], name="trial_CC")
        hev_2 = CC._complex_set

        assert hev.__contains__(he_1) is False
        assert hev.__contains__(he_5) is False
        assert hev.__contains__(he_6) is False

        assert hev_2.__contains__(he_1) is True
        assert hev_2.__contains__(he_2) is False
        assert hev_2.__contains__(he_3) is True
        assert hev_2.__contains__(he_4) is False

    def test_hyper_edge_view_repr(self):
        """Test the __repr__ method for the Hyperedge View Class."""
        he_1 = HyperEdge((1, 2, 3, 4), rank=2)

        CC = CombinatorialComplex([he_1], name="trial_CC")

        assert (
            CC._complex_set.__repr__()
            == "HyperEdgeView([(1, 2, 3, 4), (1,), (2,), (3,), (4,)])"
        )

    def test_hyper_edge_view_str(self):
        """Test the __str__ method for the Hyperedge View Class."""
        he_1 = HyperEdge((1, 2, 3, 4), rank=2)

        CC = CombinatorialComplex([he_1], name="trial_CC")

        assert (
            CC._complex_set.__str__()
            == "HyperEdgeView([(1, 2, 3, 4), (1,), (2,), (3,), (4,)])"
        )

    def test_hyper_edge_view_skeleton(self):
        """Test the skeleton method for the Hyperedge View Class."""
        he_1 = HyperEdge((1, 2, 3, 4), rank=2)

        CC = CombinatorialComplex([he_1], name="trial_CC")
        hev = CC._complex_set

        assert hev.skeleton(rank=0) == [
            frozenset({1}),
            frozenset({2}),
            frozenset({3}),
            frozenset({4}),
        ]

        assert hev.skeleton(rank=2) == [frozenset({1, 2, 3, 4})]

        assert hev.skeleton(rank=3) == []

        assert hev.skeleton(rank=0, level="uppereq") == [
            frozenset({1}),
            frozenset({2}),
            frozenset({3}),
            frozenset({4}),
            frozenset({1, 2, 3, 4}),
        ]

        assert hev.skeleton(rank=2, level="uppereq") == [frozenset({1, 2, 3, 4})]

        assert hev.skeleton(rank=3, level="uppereq") == []

        assert hev.skeleton(rank=0, level="upeq") == [
            frozenset({1}),
            frozenset({2}),
            frozenset({3}),
            frozenset({4}),
            frozenset({1, 2, 3, 4}),
        ]

        assert hev.skeleton(rank=2, level="upeq") == [frozenset({1, 2, 3, 4})]

        assert hev.skeleton(rank=3, level="upeq") == []

        assert hev.skeleton(rank=0, level="downeq") == [
            frozenset({1}),
            frozenset({2}),
            frozenset({3}),
            frozenset({4}),
        ]

        assert hev.skeleton(rank=2, level="downeq") == [
            frozenset({1}),
            frozenset({2}),
            frozenset({3}),
            frozenset({4}),
            frozenset({1, 2, 3, 4}),
        ]

        assert hev.skeleton(rank=0, level="lowereq") == [
            frozenset({1}),
            frozenset({2}),
            frozenset({3}),
            frozenset({4}),
        ]

        assert hev.skeleton(rank=2, level="lowereq") == [
            frozenset({1}),
            frozenset({2}),
            frozenset({3}),
            frozenset({4}),
            frozenset({1, 2, 3, 4}),
        ]

        with pytest.raises(TopoNetXError) as exp_exception:
            hev.skeleton(rank=2, level="should_raise_error")

        assert (
            str(exp_exception.value)
            == "level must be None, equal, 'uppereq', 'lowereq', 'upeq', 'downeq', 'uppereq', 'lower', 'up', or 'down'  "
        )

    def test_hyper_edge_view_get_rank(self):
        """Test the get_rank method of the Hyperedge View Class."""
        he_1 = HyperEdge((1, 2, 3, 4), rank=2)

        CC = CombinatorialComplex([he_1], name="trial_CC")
        hev = CC._complex_set

        assert hev.get_rank([]) == 0

        assert hev.get_rank([1]) == 0

        assert hev.get_rank([1, 2, 3, 4]) == 2

        with pytest.raises(KeyError) as exp_exception:
            hev.get_rank([1, 4, 5, 8])

        assert (
            str(exp_exception.value) == "'hyperedge [1, 4, 5, 8] is not in the complex'"
        )

        he_2 = HyperEdge({}, rank=1)

        assert hev.get_rank(he_2) == 0

        he_3 = HyperEdge({1}, rank=0)

        assert hev.get_rank(he_3) == 0

        assert hev.get_rank(he_1) == 2

        he_4 = HyperEdge({8, 1, 9, 7})

        with pytest.raises(KeyError) as exp_exception:
            hev.get_rank(he_4)

        assert (
            str(exp_exception.value)
            == f"'hyperedge {he_4.elements} is not in the complex'"
        )

        assert hev.get_rank(1) == 0
        assert hev.get_rank(2) == 0

        with pytest.raises(KeyError) as exp_exception:
            hev.get_rank(6)

        assert (
            str(exp_exception.value)
            == "'hyperedge frozenset({6}) is not in the complex'"
        )

        he_5 = HyperEdge(("1", "2", "3", "4"), rank=2)

        CC_2 = CombinatorialComplex([he_5], name="trial_CC")
        hev_2 = CC_2._complex_set

        with pytest.raises(KeyError) as exp_exception:
            hev_2.get_rank("6")

        assert (
            str(exp_exception.value)
            == "\"hyperedge frozenset({'6'}) is not in the complex\""
        )

        assert hev_2.get_rank("1") == 0
        assert hev_2.get_rank("2") == 0

    def test_hyper_edge_view_get_lower_rank(self):
        """Test the _get_lower_rank method of the Hyperedge View Class."""
        CC = CombinatorialComplex([], name="trial_CC")
        hev = CC._complex_set

        assert hev._get_lower_rank(rank=1) == -1

        he_1 = HyperEdge((1, 2, 3, 4), rank=2)
        he_2 = HyperEdge((1, 2, 3, 4, 6, 7), rank=3)

        CC = CombinatorialComplex([he_1, he_2], name="trial_CC")
        hev = CC._complex_set

        assert hev._get_lower_rank(rank=3) == -1

        assert hev._get_lower_rank(rank=0) == -1

        assert hev._get_lower_rank(rank=5) == -1

        assert hev._get_lower_rank(rank=2) == 0

        with pytest.raises(ValueError) as e:
            hev._get_lower_rank(rank=1)

        assert str(e.value) == "1 is not in list"

    def test_hyper_edge_view_get_higher_rank(self):
        """Test the _get_higher_rank method of the Hyperedge View Class."""
        CC = CombinatorialComplex([], name="trial_CC")
        hev = CC._complex_set

        assert hev._get_higher_rank(rank=1) == -1

        he_1 = HyperEdge((1, 2, 3, 4), rank=2)
        he_2 = HyperEdge((1, 2, 3, 4, 6, 7), rank=3)

        CC = CombinatorialComplex([he_1, he_2], name="trial_CC")
        hev = CC._complex_set

        assert hev._get_higher_rank(rank=3) == -1

        assert hev._get_higher_rank(rank=0) == -1

        assert hev._get_higher_rank(rank=6) == -1

        assert hev._get_higher_rank(rank=2) == 3

        with pytest.raises(ValueError) as e:
            hev._get_higher_rank(rank=1)

        assert str(e.value) == "1 is not in list"
