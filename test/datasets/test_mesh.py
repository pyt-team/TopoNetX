"""Test mesh dataset."""

import pytest

from toponetx.datasets.mesh import coseg, shrec_16, stanford_bunny


class TestMeshDatasets:
    """Test datasets utils."""

    def test_stanford_bunny(self):
        """Test stanford_bunny."""
        simplicialbunny = stanford_bunny("simplicial")

        assert len(simplicialbunny) == 2503

        cellbunny = stanford_bunny("cell")

        assert len(cellbunny) == 2503

        with pytest.raises(ValueError):
            stanford_bunny("polyhedral")

    def test_shrec_16_small(self):
        """Test shrec_16."""
        shrec_training, shrec_testing = shrec_16(size="small")

        assert len(shrec_training["complexes"]) == 100
        assert len(shrec_testing["complexes"]) == 20
        assert len(shrec_training["label"]) == 100
        assert len(shrec_testing["label"]) == 20
        assert len(shrec_training["node_feat"]) == 100
        assert len(shrec_testing["node_feat"]) == 20
        assert len(shrec_training["edge_feat"]) == 100
        assert len(shrec_testing["edge_feat"]) == 20
        assert len(shrec_training["face_feat"]) == 100
        assert len(shrec_testing["face_feat"]) == 20

    def test_shrec_16_full(self):
        """Test shrec_16."""
        shrec_training, shrec_testing = shrec_16(size="full")

        assert len(shrec_training["complexes"]) == 480
        assert len(shrec_testing["complexes"]) == 120
        assert len(shrec_training["label"]) == 480
        assert len(shrec_testing["label"]) == 120
        assert len(shrec_training["node_feat"]) == 480
        assert len(shrec_testing["node_feat"]) == 120
        assert len(shrec_training["edge_feat"]) == 480
        assert len(shrec_testing["edge_feat"]) == 120
        assert len(shrec_training["face_feat"]) == 480
        assert len(shrec_testing["face_feat"]) == 120

    def test_shrec_16_invalid(self):
        """Test shrec_16."""
        with pytest.raises(ValueError):
            shrec_16(size="huge")

    def test_coseg(self):
        """Test coseg."""
        coseg_data = coseg(data="alien")

        assert len(coseg_data["complexes"]) != 0
        assert len(coseg_data["node_feat"]) != 0
        assert len(coseg_data["edge_feat"]) != 0
        assert len(coseg_data["face_feat"]) != 0
        assert len(coseg_data["face_label"]) != 0

        nodes, edges, faces = coseg_data["complexes"][0].shape
        n_nodes = coseg_data["node_feat"][0].shape
        n_faces = coseg_data["face_feat"][0].shape
        assert nodes == n_nodes[0]
        assert faces == n_faces[0]
        assert len(coseg_data["face_label"][0]) == faces

        with pytest.raises(ValueError):
            coseg(data="cars")
