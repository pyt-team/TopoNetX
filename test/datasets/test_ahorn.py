"""Tests for AHORN dataset loader."""
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false

from io import StringIO

import networkx as nx
import pytest

from toponetx.datasets.ahorn import load_ahorn_dataset, read_ahorn_dataset

try:
    import ahorn_loader
except ImportError:
    ahorn_loader = None


@pytest.mark.skipif(
    ahorn_loader is None, reason="Optional dependency `ahorn-loader` not installed."
)
def test_load_karate_club_matches_networkx_sc() -> None:
    """Load AHORN 'karate-club' and compare with networkx.karate_club_graph."""
    SC = load_ahorn_dataset("karate-club")

    # Build graph from simplicial complex
    G_sc = SC.graph_skeleton()
    nodes_sc = set(G_sc.nodes())
    edges_sc = {tuple(sorted(e)) for e in G_sc.edges()}

    # Reference networkx graph
    G_nx = nx.karate_club_graph()
    nodes_nx = set(G_nx.nodes())
    edges_nx = {tuple(sorted(e)) for e in G_nx.edges()}

    # Basic structure equality
    assert nodes_sc == nodes_nx
    assert edges_sc == edges_nx


@pytest.mark.skipif(
    ahorn_loader is not None, reason="Optional dependency `ahorn-loader` installed."
)
def test_load_missing_dependency_raises() -> None:
    """When `ahorn-loader` is not installed, load_ahorn_dataset raises RuntimeError."""
    with pytest.raises(RuntimeError, match=r"optional `ahorn-loader`"):
        load_ahorn_dataset("karate-club")


@pytest.mark.skipif(
    ahorn_loader is not None, reason="Optional dependency `ahorn-loader` installed."
)
def test_read_missing_dependency_raises() -> None:
    """When `ahorn-loader` is not installed, read_ahorn_dataset raises RuntimeError."""
    with pytest.raises(RuntimeError, match=r"optional `ahorn-loader`"):
        read_ahorn_dataset("dummy.json")


@pytest.mark.skipif(
    ahorn_loader is None, reason="Optional dependency `ahorn-loader` not installed."
)
def test_read_multi_network_dataset() -> None:
    """Test reading a multi-network AHORN dataset from a mock file."""
    mock_data = """{"name": "Mock Multi-Network Dataset"}
{"id": "network-001"}
0 {"label": "node0_net1"}
1 {"label": "node1_net1"}
2 {"label": "node2_net1"}
{"id": "network-002"}
0,1 {"weight": 2.0}
1,2 {"weight": 3.0}
0,1,2 {}
"""

    result = read_ahorn_dataset(StringIO(mock_data))

    assert isinstance(result, list)
    assert len(result) == 2

    assert len(list(result[0].simplices)) == 3
    assert result[0].complex["id"] == "network-001"
    assert result[0].nodes[0]["label"] == "node0_net1"
    assert result[0].nodes[1]["label"] == "node1_net1"
    assert result[0].nodes[2]["label"] == "node2_net1"

    assert len(list(result[1].simplices)) == 7
    assert result[1].complex["id"] == "network-002"
    assert result[1].simplices[(0, 1)]["weight"] == 2.0
    assert result[1].simplices[(1, 2)]["weight"] == 3.0
