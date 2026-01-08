"""Test graph dataset."""

from toponetx.datasets.graph import coauthorship, karate_club


class TestGraph:
    """Test datasets utils."""

    def test_karate_club(self):
        """Test karate_club."""
        simplicial_karate_club_data = karate_club("simplicial")

        assert len(simplicial_karate_club_data.get_simplex_attributes("node_feat")) != 0
        assert len(simplicial_karate_club_data.get_simplex_attributes("edge_feat")) != 0
        assert len(simplicial_karate_club_data.get_simplex_attributes("face_feat")) != 0
        assert (
            len(simplicial_karate_club_data.get_simplex_attributes("tetrahedron_feat"))
            != 0
        )

        cell_karate_club_data = karate_club("cell")

        assert len(cell_karate_club_data.get_cell_attributes("node_feat", rank=0)) != 0
        assert len(cell_karate_club_data.get_cell_attributes("edge_feat", rank=1)) != 0
        assert len(cell_karate_club_data.get_cell_attributes("cell_feat", rank=2)) != 0

    def test_coauthorship(self):
        """Test coauthorship."""
        simplicial_coauthorship_data = coauthorship()

        assert (
            len(simplicial_coauthorship_data.get_simplex_attributes("citations")) != 0
        )
