"""Test mesh dataset."""

import unittest

from toponetx import CellComplex, SimplicialComplex
from toponetx.datasets.graph import karate_club_complex


class TestMeshDatasets(unittest.TestCase):
    """Test datasets utils."""

    def test_karate_club_complex(self):
        """Test stanford_bunny."""
        simplicial_karate_club_data = karate_club_complex("simplicial")

        assert "node_feat" in simplicial_karate_club_data
        assert "edge_feat" in simplicial_karate_club_data
        assert "face_feat" in simplicial_karate_club_data
        assert "tet_feat" in simplicial_karate_club_data

        cell_karate_club_data = karate_club_complex("cell")

        assert "node_feat" in cell_karate_club_data
        assert "edge_feat" in cell_karate_club_data
        assert "cell_feat" in cell_karate_club_data


if __name__ == "__main__":
    unittest.main()
