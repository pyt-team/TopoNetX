import unittest

import networkx as nx

from toponetx.classes.abstract_cell import AbstractCell
from toponetx.classes.combinatorial_complex import CombinatorialComplex


class TestCombinatorialComplex(unittest.TestCase):
    def create_CombinatorialComplex_from_nodes_and_cells(self):
        # create a collection of Node objects and a collection of DynamicCells
        # and pass them to DynamicCombinatorialComplex

        y1 = AbstractCell(elements=[1, 2], rank=1)
        y2 = AbstractCell(elements=[2, 4], rank=1)
        y3 = AbstractCell(elements=[3, 5], rank=1)
        y4 = AbstractCell(elements=[4, 5], rank=1)
        y5 = AbstractCell(elements=[5, 7], rank=1)
        # define the DynamicCombinatorialComplex from a list of cells
        CC = CombinatorialComplex(cells=[y1, y2, y3, y4, y5])

        assert y1 in CC.cells
        assert y2 in CC.cells
        assert y3 in CC.cells
        assert y4 in CC.cells
        assert y5 in CC.cells

    def create_CombinatorialComplex_from_graph(self):

        G = nx.Graph()  # networkx graph
        G.add_edge(0, 1)
        G.add_edge(0, 3)
        G.add_edge(0, 4)
        G.add_edge(1, 4)
        CC = CombinatorialComplex(cells=G)
        assert (0, 1) in CC.cells
        assert (0, 3) in CC.cells
        assert (0, 4) in CC.cells
        assert (1, 4) in CC.cells
        assert (0, 5) not in CC.cells


if __name__ == "__main__":
    unittest.main()
