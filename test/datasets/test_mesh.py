"""Test mesh dataset."""

import unittest

from toponetx import CellComplex, SimplicialComplex
from toponetx.datasets.mesh import shrec_16, stanford_bunny


class TestMeshDatasets(unittest.TestCase):
    """Test datasets utils."""

    def test_stanford_bunny(self):
        """Test stanford_bunny."""
        simplicialbunny = stanford_bunny("simplicial complex")

        assert len(simplicialbunny) == 2503

        cellbunny = stanford_bunny("cell complex")

        assert len(cellbunny) == 2503

        with self.assertRaises(ValueError):
            stanford_bunny("polyhedral complex")

    def test_shrec_16(self):
        """Test shrec_16."""
        shrec_training, shrec_testing = shrec_16()

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


if __name__ == "__main__":
    unittest.main()
