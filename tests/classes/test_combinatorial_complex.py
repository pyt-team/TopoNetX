import unittest

from toponetx.classes.combinatorial_complex import CombinatorialComplex
from toponetx.classes.ranked_entity import (
    CellObject,
    Node,
    RankedEntity,
    RankedEntitySet,
)


class TestCombintorialComplex(unittest.TestCase):
    def test_combinatorial_complex_ranks(self):
        x1 = Node(1)
        x2 = Node(2)
        y1 = CellObject(elements=[x1, x2], rank=1)
        self.assertEqual(x1.rank, 0)
        self.assertEqual(x2.rank, 0)
        self.assertEqual(y1.rank, 1)

    def test_combinatorial_complex_skeleton(self):
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
        CC = CombinatorialComplex(cells=E)
        self.assertEqual(len(CC.skeleton(0)), 5)
        self.assertEqual(len(CC.skeleton(1)), 6)
        self.assertEqual(len(CC.skeleton(2)), 1)


if __name__ == "__main__":
    unittest.main()
