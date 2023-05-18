"""Test Cell class."""

import unittest
from collections import defaultdict

from toponetx.classes.cell_complex import CellComplex
from toponetx.utils.structure import (
    neighborhood_list_to_neighborhood_dict,
    sparse_array_to_neighborhood_dict,
    sparse_array_to_neighborhood_list,
)


class TestStructure(unittest.TestCase):
    def test_sparse_array_to_neighborhood_list(self):
        """Test the sparse_array_to_neighborhood_list function."""
        c = CellComplex()
        c.add_cells_from([[1, 2, 3], [4, 5, 6]], rank=2)
        row, col, B1 = c.incidence_matrix(1, index=True)
        output = sparse_array_to_neighborhood_list(B1)
        expected = [
            (0, 0),
            (1, 0),
            (0, 1),
            (2, 1),
            (1, 2),
            (2, 2),
            (3, 3),
            (4, 3),
            (3, 4),
            (5, 4),
            (4, 5),
            (5, 5),
        ]

        assert list(output) == expected

        output1 = sparse_array_to_neighborhood_list(
            B1, list(col.keys()), list(row.keys())
        )

        expected1 = [
            (1, (1, 2)),
            (2, (1, 2)),
            (1, (1, 3)),
            (3, (1, 3)),
            (2, (2, 3)),
            (3, (2, 3)),
            (4, (4, 5)),
            (5, (4, 5)),
            (4, (4, 6)),
            (6, (4, 6)),
            (5, (5, 6)),
            (6, (5, 6)),
        ]

        assert list(output1) == expected1

    def test_neighborhood_list_to_neighborhood_dict(self):
        """
        Test the neighborhood_list_to_neighborhood_dict function.
        """
        neighborhood_list = [
            (0, 0),
            (1, 0),
            (0, 1),
            (2, 1),
            (1, 2),
            (2, 2),
            (3, 3),
            (4, 3),
            (3, 4),
            (5, 4),
            (4, 5),
            (5, 5),
        ]

        out = neighborhood_list_to_neighborhood_dict(neighborhood_list)

        d = defaultdict(
            list, {0: [0, 1], 1: [0, 2], 2: [1, 2], 3: [3, 4], 4: [3, 5], 5: [4, 5]}
        )
        assert out == d

    def test_sparse_array_to_neighborhood_dict(self):
        """
        Test the sparse_array_to_neighborhood_dict function.
        """
        c = CellComplex()
        c.add_cells_from([[1, 2, 3], [4, 5, 6]], rank=2)
        row, col, B1 = c.incidence_matrix(1, index=True)
        output = sparse_array_to_neighborhood_dict(B1)

        d = defaultdict(
            list, {0: [0, 1], 1: [0, 2], 2: [1, 2], 3: [3, 4], 4: [3, 5], 5: [4, 5]}
        )
        assert output == d


if __name__ == "__main__":
    unittest.main()
