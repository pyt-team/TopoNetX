import unittest

import networkx as nx

from toponetx.classes.dynamic_cell import DynamicCell
from toponetx.classes.dynamic_combinatorial_complex import DynamicCombinatorialComplex
from toponetx.classes.node import Node
from toponetx.classes.ranked_entity import RankedEntity, RankedEntitySet


class TestDynamicCombinatorialComplex(unittest.TestCase):
    def test_dynamic_combinatorial_complex_ranks(self):
        x1 = Node(1)
        x2 = Node(2)
        y1 = DynamicCell(elements=[x1, x2], rank=1)
        self.assertEqual(x1.rank, 0)
        self.assertEqual(x2.rank, 0)
        self.assertEqual(y1.rank, 1)

    def test_dynamic_combinatorial_complex_skeleton(self):
        x1 = RankedEntity("x1", rank=0)
        x2 = RankedEntity("x2", rank=0)
        x3 = RankedEntity("x3", rank=0)
        x4 = RankedEntity("x4", rank=0)
        x5 = RankedEntity("x5", rank=0)
        y1 = RankedEntity("y1", [x1, x2], rank=1)
        y2 = RankedEntity("y2", [x2, x3], rank=1)
        y3 = RankedEntity("y3", [x3, x4], rank=1)
        y4 = RankedEntity("y4", [x4, x1], rank=1)
        y5 = RankedEntity("y5", [x4, x5], rank=1)
        y6 = RankedEntity("y6", [x4, x5], rank=1)
        w = RankedEntity("w", [x4, x5, x1], rank=2)
        E = RankedEntitySet("E", [y1, y2, y3, y4, y5, w, y6])
        CC = DynamicCombinatorialComplex(cells=E)
        self.assertEqual(len(CC.skeleton(0)), 5)
        self.assertEqual(len(CC.skeleton(1)), 6)
        self.assertEqual(len(CC.skeleton(2)), 1)

    def create_DynamicCombinatorialComplex_from_nodes_and_cells(self):
        # create a collection of Node objects and a collection of DynamicCells
        # and pass them to DynamicCombinatorialComplex
        x1 = Node(1)
        x2 = Node(2)
        x3 = Node(3)
        x4 = Node(4)
        x5 = Node(5)
        y1 = DynamicCell(elements=[x1, x2], rank=1)
        y2 = DynamicCell(elements=[x2, x3], rank=1)
        y3 = DynamicCell(elements=[x3, x4], rank=1)
        y4 = DynamicCell(elements=[x4, x1], rank=1)
        y5 = DynamicCell(elements=[x4, x5], rank=1)
        w = DynamicCell(elements=[x4, x5, x1], rank=2)
        # define the DynamicCombinatorialComplex from a list of cells
        CC = DynamicCombinatorialComplex(cells=[y1, y2, y3, y4, y5, w])

        assert y1 in CC.cells
        assert y2 in CC.cells
        assert y3 in CC.cells
        assert y4 in CC.cells
        assert y5 in CC.cells
        assert w in CC.cells

        assert x1 in CC
        assert x2 in CC

    def create_DynamicCombinatorialComplex_from_graph(self):

        G = nx.Graph()  # networkx graph
        G.add_edge(0, 1)
        G.add_edge(0, 3)
        G.add_edge(0, 4)
        G.add_edge(1, 4)
        CC = DynamicCombinatorialComplex(cells=G)
        assert (0, 1) in CC.cells
        assert (0, 3) in CC.cells
        assert (0, 4) in CC.cells
        assert (1, 4) in CC.cells


if __name__ == "__main__":
    unittest.main()
