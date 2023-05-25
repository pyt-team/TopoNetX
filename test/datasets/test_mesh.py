"""Test mesh dataset."""

import unittest

from toponetx import CellComplex, SimplicialComplex
from toponetx.datasets.mesh import stanford_bunny


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


if __name__ == "__main__":
    unittest.main()
