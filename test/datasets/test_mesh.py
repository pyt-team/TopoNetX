"""Test mesh dataset."""

import unittest

from toponetx import CellComplex, SimplicialComplex
from toponetx.datasets.mesh import shrec_16, stanford_bunny


class TestMeshDatasets(unittest.TestCase):
    """Test datasets utils."""

    def test_stanford_bunny(self):
        """Test stanford_bunny."""
        simplicialbunny = stanford_bunny("simplicial")

        assert len(simplicialbunny) == 2503

        cellbunny = stanford_bunny("cell")

        assert len(cellbunny) == 2503

        with self.assertRaises(ValueError):
            stanford_bunny("polyhedral")

    def test_shrec_16(self):
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


if __name__ == "__main__":
    unittest.main()
