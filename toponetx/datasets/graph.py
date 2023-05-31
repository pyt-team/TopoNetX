from torch_geometric.utils.convert import to_networkx

import toponetx as tnx


class GraphDataset:
    def __init__(self, dataset, directed=False, domain="cellular", max_dim=3):
        self.graph = dataset

        if domain == "simplicial":
            self.domain = self.to_simplicial(directed, max_dim)

    def to_simplicial(self, directed, max_dim):
        networkx_graph = to_networkx(self.graph)
        if directed:
            networkx_graph = networkx_graph.to_undirected()

        self.domain = tnx.transform.graph_2_clique_complex(
            networkx_graph, max_dim=max_dim
        )

    # def to_cellular(self, directed):
    #     pass


# dataset = GraphDataset(KarateClub(), directed=True, domain="cellular")
# dataset.graph
# dataset.domain
