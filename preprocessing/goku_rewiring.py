from .graph_recover import recover_latent_graph
from .graph_sparsification import sparsification
import torch


def goku(edge_list, features, num_relations=2, k_guess=10, method='kts', epsilon=0.1, step_size=0.1, device='cuda:0',
         to_undirected=True, metric='degree', beta=1.0):
    num_nodes = features.shape[0]
    edge_index = torch.tensor(edge_list.T, dtype=torch.long).to(device) # (2, E)
    m = edge_index.shape[1]

    # apply graph densification
    latent_edge_index = recover_latent_graph(edge_index, num_nodes, k_guess, step_size, metric)


    k = latent_edge_index.size(1)

    edge_weight = torch.ones_like(latent_edge_index[0]).cpu() * m / k

    # apply graph sparsification
    computational_edge_index, edge_type, computational_edge_weight = sparsification(edge_index, latent_edge_index.cpu().numpy().transpose(),
                                                         edge_weight, features, num_samples=m, method=method, epsilon=epsilon,
                                                         undirected=to_undirected, num_relations=num_relations, beta=beta
                                                                                    )


    return computational_edge_index, edge_type, computational_edge_weight


