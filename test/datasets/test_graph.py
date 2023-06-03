"""Test graph dataset."""

from toponetx import CellComplex, SimplicialComplex
from toponetx.datasets.graph import karate_club


class TestGraph:
    """Test datasets utils."""

    def test_karate_club(self):
        """Test karate_club."""
        simplicial_karate_club_data = karate_club("simplicial")

        assert "node_feat" in simplicial_karate_club_data
        assert "edge_feat" in simplicial_karate_club_data
        assert "face_feat" in simplicial_karate_club_data
        assert "tetrahedron_feat" in simplicial_karate_club_data

        cell_karate_club_data = karate_club("cell")

        assert "node_feat" in cell_karate_club_data
        assert "edge_feat" in cell_karate_club_data
        assert "cell_feat" in cell_karate_club_data
