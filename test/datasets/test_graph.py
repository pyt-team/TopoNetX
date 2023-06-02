"""Test graph dataset."""

import unittest

from toponetx import CellComplex, SimplicialComplex
from toponetx.datasets.graph import karate_club


class TestGraph(unittest.TestCase):
    """Test datasets utils."""

    def test_karate_club(self):
        """Test stanford_bunny."""
        simplicial_karate_club_data = karate_club("simplicial")

        assert "node_feat" in simplicial_karate_club_data
        assert "edge_feat" in simplicial_karate_club_data
        assert "face_feat" in simplicial_karate_club_data
        assert "tet_feat" in simplicial_karate_club_data

        cell_karate_club_data = karate_club("cell")

        assert "node_feat" in cell_karate_club_data
        assert "edge_feat" in cell_karate_club_data
        assert "cell_feat" in cell_karate_club_data


if __name__ == "__main__":
    unittest.main()
