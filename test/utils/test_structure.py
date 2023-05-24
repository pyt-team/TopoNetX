"""Test Cell class."""

import unittest
from collections import defaultdict

from scipy.sparse import csr_matrix

from toponetx.classes.cell_complex import CellComplex
from toponetx.utils.structure import (
    incidence_to_adjacency,
    neighborhood_list_to_neighborhood_dict,
    sparse_array_to_neighborhood_dict,
    sparse_array_to_neighborhood_list,
)


class TestStructure(unittest.TestCase):
    """Test structure module."""

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

        with self.assertRaises(ValueError):
            output1 = sparse_array_to_neighborhood_list(B1, list(row.keys()))

    def test_neighborhood_list_to_neighborhood_dict(self):
        """Test the neighborhood_list_to_neighborhood_dict function."""
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

    def test_neighborhood_list_to_neighborhood_with_dict(self):
        """Test that neighborhood_list_to_neighborhood_dict works correctly with specified node dictionary."""
        node_dict = {1: "a", 2: "b", 3: "c"}
        neigh_list = [(1, 2), (2, 1), (2, 3), (3, 2)]
        neigh_dict = neighborhood_list_to_neighborhood_dict(
            neigh_list, node_dict, node_dict
        )
        assert len(neigh_dict) == 3
        assert neigh_dict["a"] == ["b"]
        assert set(neigh_dict["b"]) == {"a", "c"}  # order irrelevant
        assert neigh_dict["c"] == ["b"]

    def test_sparse_array_to_neighborhood_dict(self):
        """Test the sparse_array_to_neighborhood_dict function."""
        c = CellComplex()
        c.add_cells_from([[1, 2, 3], [4, 5, 6]], rank=2)
        row, col, B1 = c.incidence_matrix(1, index=True)
        output = sparse_array_to_neighborhood_dict(B1)

        d = defaultdict(
            list, {0: [0, 1], 1: [0, 2], 2: [1, 2], 3: [3, 4], 4: [3, 5], 5: [4, 5]}
        )
        assert output == d

    def test_incidence_to_adjacency(self):
        """Test incidence to adjacency.

        uses transposed of cell-edge incidence for cell complex [(1,2,3,4), (1,2,4,3)]
        """
        incidence = csr_matrix(
            [[1, 0, -1, 1, 0, 1], [1, -1, 0, 0, 1, -1]]
        )  # already transposed (check for upper adj.)
        adj = incidence_to_adjacency(incidence)
        expected_adj = csr_matrix(
            [
                [
                    0,
                    1,
                    1,
                    1,
                    1,
                    2,
                ],  # (1,2) and (3,4) are upper-adjacent in both 2-cells
                [1, 0, 0, 0, 1, 1],
                [1, 0, 0, 1, 0, 1],
                [1, 0, 1, 0, 0, 1],
                [1, 1, 0, 0, 0, 1],
                [2, 1, 1, 1, 1, 0],
            ]
        )
        assert (
            adj != expected_adj
        ).nnz == 0  # tests sparsity of difference -> if the difference has no non-zero entries, it is the same


if __name__ == "__main__":
    unittest.main()
