"""Test the random simplicial complex generators."""

from toponetx.generators.random_simplicial_complexes import (
    linial_meshulam_complex,
    multiparameter_linial_meshulam_complex,
    random_clique_complex,
)


class TestLinialMeshulamComplex:
    """Test the `linial_meshulam_complex` function."""

    def test_zero_nodes(self):
        """Test `linial_meshulam_complex` for zero nodes."""
        SC = linial_meshulam_complex(0, 0.1)
        assert SC.shape == ()

    def test_zero_probability(self):
        """Test `linial_meshulam_complex` for zero probability."""
        SC = linial_meshulam_complex(10, 0.0)
        assert SC.shape == (10, 45)

    def test_one_probability(self):
        """Test `linial_meshulam_complex` for one probability."""
        SC = linial_meshulam_complex(10, 1.0)
        assert SC.shape == (10, 45, 120)


class TestRandomCliqueComplex:
    """Test the `random_clique_complex` function."""

    def test_zero_nodes(self):
        """Test `random_clique_complex` for zero nodes."""
        SC = random_clique_complex(0, 0.1)
        assert SC.shape == ()

    def test_zero_probability(self):
        """Test `random_clique_complex` for zero probability."""
        SC = random_clique_complex(10, 0.0)
        assert SC.shape == (10,)


class TestMultiparameterLinialMeshulamComplex:
    """Test the `multiparameter_linial_meshulam_complex` function."""

    def test_zero_nodes(self):
        """Test `multiparameter_linial_meshulam_complex` for zero nodes."""
        SC = multiparameter_linial_meshulam_complex(0, [1.0, 0.1])
        assert SC.shape == ()

    def test_one_probabilities(self):
        """Test `multiparameter_linial_meshulam_complex` for one probabilities."""
        SC = multiparameter_linial_meshulam_complex(10, [1.0])
        assert SC.shape == (10, 45)

        SC = multiparameter_linial_meshulam_complex(10, [1.0, 1.0])
        assert SC.shape == (10, 45, 120)

    def test_ignore_higher_cliques(self):
        """Test that `multiparameter_linial_meshulam_complex` ignores higher-order cliques.

        It is crucial that higher-order cliques are only added when all lower-order
        cliques have been added as well. Otherwise, the probabilities are not correct.
        For this, we artificially disallow triangles but force 4-cliques with
        probability one. Nonetheless, no 3-simplex should be present.
        """
        SC = multiparameter_linial_meshulam_complex(10, [1.0, 0.0, 1.0])
        assert SC.shape == (10, 45)
