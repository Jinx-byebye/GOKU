from .graph_recover import recover_latent_graph
from .graph_sparsification import sparsification
import torch


def goku(edge_list, features, num_relations=2, mini_k=10, method='kts', epsilon=0.1, step_size=0.1, device='cuda:0',
         to_undirected=True, metric='degree', beta=1.0):
    """
    Graph Optimization through Kombinatorial Unification (GOKU) rewiring algorithm.
    
    This method combines graph densification (latent graph recovery) with sparsification to 
    create a computational graph structure that preserves key properties of the original graph
    while improving message passing capabilities for GNN architectures.
    
    Args:
        edge_list (numpy.ndarray): Original edge list of shape (E, 2).
        features (torch.Tensor): Node feature matrix of shape (num_nodes, feature_dim).
        num_relations (int): Number of relation types to use in the computational graph.
        mini_k (int): Initial guess for the number of edges to add during densification.
        method (str): Method for effective resistance calculation ('ext', 'spl', 'kts').
        epsilon (float): Accuracy parameter for effective resistance calculation.
        step_size (float): Step size for increasing epsilon during latent graph recovery.
        device (str): Device to use for calculations ('cpu' or 'cuda:0').
        to_undirected (bool): Whether to convert the sampled graph to an undirected graph.
        metric (str): Method for selecting edges to add ('degree' or other).
        beta (float): Scaling factor for sparsification calculations.
        
    Returns:
        tuple: (computational_edge_index, edge_type, computational_edge_weight)
            - computational_edge_index: Tensor of shape (2, E') representing rewired edges
            - edge_type: Tensor of shape (E',) representing edge types
            - computational_edge_weight: Tensor of shape (E',) representing edge weights
    """
    num_nodes = features.shape[0]
    edge_index = torch.tensor(edge_list.T, dtype=torch.long).to(device)  # Convert to tensor of shape (2, E)
    original_edge_count = edge_index.shape[1]

    # Step 1: Apply graph densification to recover latent graph structure
    latent_edge_index = recover_latent_graph(
        edge_index, 
        num_nodes, 
        mini_k,
        step_size, 
        metric
    )

    latent_edge_count = latent_edge_index.size(1)

    # Create uniform edge weights, scaled to maintain the original average weight
    edge_weight = torch.ones_like(latent_edge_index[0]).cpu() * original_edge_count / latent_edge_count

    # Step 2: Apply graph sparsification to create the computational graph
    computational_edge_index, edge_type, computational_edge_weight = sparsification(
        edge_index,  # Original edges
        latent_edge_index.cpu().numpy().transpose(),  # Latent edges
        edge_weight,  # Uniform weights
        features,  # Node features
        num_samples=original_edge_count,  # Target number of edges in computational graph
        method=method,  
        epsilon=epsilon,
        undirected=to_undirected,
        num_relations=num_relations,
        beta=beta
    )

    return computational_edge_index, edge_type, computational_edge_weight


