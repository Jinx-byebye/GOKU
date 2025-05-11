import numpy as np
from scipy import sparse
from scipy.sparse.linalg import cg
from numba import njit, prange
import torch
from torch.utils.data import WeightedRandomSampler
from torch_geometric.utils import to_undirected
from typing import Optional, Tuple


# torch_scatter/utils.py
def broadcast(src: torch.Tensor, other: torch.Tensor, dim: int):
    """
    Broadcast a tensor to match the shape of another tensor along a specified dimension.
    
    Args:
        src: Source tensor to broadcast
        other: Target tensor whose shape will be matched
        dim: Dimension along which to broadcast
        
    Returns:
        Broadcasted tensor with shape matching other
    """
    if dim < 0:
        dim = other.dim() + dim
    if src.dim() == 1:
        for _ in range(0, dim):
            src = src.unsqueeze(0)
    for _ in range(src.dim(), other.dim()):
        src = src.unsqueeze(-1)
    src = src.expand(other.size())
    return src


# torch_scatter/scatter.py
def scatter_sum_raw(
    src: torch.Tensor,
    index: torch.Tensor,
    dim: int = -1,
    out: Optional[torch.Tensor] = None,
    dim_size: Optional[int] = None,
) -> torch.Tensor:
    """
    Sums all values from src tensor into out at the indices specified in index.
    
    Args:
        src: Source tensor containing values to scatter
        index: Indices indicating where to scatter the values
        dim: Dimension along which to scatter
        out: Optional output tensor
        dim_size: Size of the output dimension
        
    Returns:
        Tensor with scattered sum values
    """
    index = broadcast(index, src, dim)
    if out is None:
        size = list(src.size())
        if dim_size is not None:
            size[dim] = dim_size
        elif index.numel() == 0:
            size[dim] = 0
        else:
            size[dim] = int(index.max()) + 1
        out = torch.zeros(size, dtype=src.dtype, device=src.device)
        return out.scatter_add_(dim, index, src)
    else:
        return out.scatter_add_(dim, index, src)


# torch_scatter/scatter.py
def scatter_add_raw(
    src: torch.Tensor,
    index: torch.Tensor,
    dim: int = -1,
    out: Optional[torch.Tensor] = None,
    dim_size: Optional[int] = None,
) -> torch.Tensor:
    """
    Alias for scatter_sum_raw.
    """
    return scatter_sum_raw(src, index, dim, out, dim_size)


@njit(nopython=True)
def adjacency_to_edge_list(adjacency_matrix):
    """
    Convert an adjacency matrix to an edge list.

    Args:
        adjacency_matrix: Adjacency matrix representation of a graph

    Returns:
        tuple: Edge list and corresponding weights
    """
    # Get non-zero edges from the upper triangle (undirected graph)
    j, i = np.nonzero(np.triu(adjacency_matrix))
    edge_list = np.vstack((i, j))
    # Extract edge weights from the adjacency matrix
    weights = adjacency_matrix[np.triu(adjacency_matrix) != 0]

    return edge_list.transpose(), weights

@njit(nopython=True)
def edge_list_to_adjacency(edge_list, weights):
    """
    Convert an edge list to an adjacency matrix.

    Args:
        edge_list: List of edges where each row is (source, target)
        weights: Corresponding edge weights

    Returns:
        ndarray: Adjacency matrix
    """
    n = np.max(edge_list) + 1  # Determine number of nodes
    adjacency_matrix = np.zeros(shape=(n, n))

    # Populate the adjacency matrix (symmetrically for undirected graph)
    for i in range(np.shape(edge_list)[0]):
        n1, n2 = edge_list[i, :]
        w = weights[i]
        adjacency_matrix[n1, n2] = w
        adjacency_matrix[n2, n1] = w  # Ensure symmetry

    return adjacency_matrix


def edge_list_to_sparse_adjacency(edge_list, weights):
    """
    Convert an edge list to a sparse adjacency matrix.

    Args:
        edge_list: List of edges where each row is (source, target)
        weights: Corresponding edge weights

    Returns:
        csr_matrix: Sparse adjacency matrix in CSR format
    """
    n = np.max(edge_list) + 1  # Determine number of nodes
    # Create sparse matrix from COO format
    adjacency_matrix = sparse.csr_matrix((weights, (edge_list[:, 0], edge_list[:, 1])), shape=(n, n))
    # Make it symmetric (undirected)
    adjacency_matrix = adjacency_matrix + adjacency_matrix.transpose()

    return adjacency_matrix

@njit(nopython=True)
def compute_laplacian(adjacency_matrix):
    """
    Compute the Laplacian matrix from an adjacency matrix.

    Args:
        adjacency_matrix: Adjacency matrix of a graph

    Returns:
        ndarray: Laplacian matrix (L = D - A)
    """
    # Compute diagonal degree matrix
    degree_matrix = np.diag(np.sum(abs(adjacency_matrix), 1))
    # Laplacian is D - A
    laplacian = degree_matrix - adjacency_matrix
    return laplacian


def compute_sparse_laplacian(adjacency_matrix):
    """
    Compute the Laplacian matrix from a sparse adjacency matrix.

    Args:
        adjacency_matrix: Sparse adjacency matrix in CSR format

    Returns:
        csr_matrix: Laplacian matrix in sparse format
    """
    # Use scipy's built-in Laplacian computation for sparse graphs
    laplacian = sparse.csgraph.laplacian(adjacency_matrix)
    return laplacian


def create_signed_incidence_matrix(edge_list):
    """
    Compute the signed-edge vertex incidence matrix.

    Args:
        edge_list: List of edges where each row is (source, target)

    Returns:
        csr_matrix: Sparse vertex incidence matrix (B)
    """
    m = np.shape(edge_list)[0]  # Number of edges
    edge_list = edge_list.transpose()  # Transpose to make rows correspond to edges

    # Create incidence matrix data: +1 for source nodes, -1 for target nodes
    data = [1] * m + [-1] * m
    row_indices = list(range(0, m)) + list(range(0, m))
    col_indices = edge_list[0, :].tolist() + edge_list[1, :].tolist()

    # Build sparse matrix in CSR format
    incidence_matrix = sparse.csr_matrix((data, (row_indices, col_indices)))

    return incidence_matrix


def create_weight_diagonal_matrix(weights):
    """
    Compute the diagonal weights matrix.

    Args:
        weights: Edge weights vector

    Returns:
        dia_matrix: Diagonal matrix of square root of weights
    """
    m = len(weights)
    # Element-wise square root of weights for the diagonal
    weights_sqrt = np.sqrt(weights)
    # Create diagonal sparse matrix
    weight_matrix = sparse.dia_matrix((weights_sqrt, [0]), shape=(m, m))

    return weight_matrix


def compute_effective_resistance(edge_list, weights, epsilon, method='kts', tol=1e-10, precon=False):
    """
    Approximate effective resistance using various methods.

    Args:
        edge_list: List of edges where each row is (source, target)
        weights: List of edge weights
        epsilon: Accuracy control parameter
        method: Type of calculation ('ext' for exact, 'spl' for splicing, 'kts' for Koutis method)
        tol: Tolerance for convergence
        precon: Preconditioner for the solver

    Returns:
        ndarray: Effective resistance values for each edge
    """
    m = np.shape(edge_list)[0]  # Number of edges
    n = np.max(edge_list) + 1  # Number of nodes

    # Create graph representation matrices
    adjacency_matrix = edge_list_to_sparse_adjacency(edge_list, weights)
    laplacian = compute_sparse_laplacian(adjacency_matrix)
    incidence_matrix = create_signed_incidence_matrix(edge_list)
    weight_matrix = create_weight_diagonal_matrix(weights)
    
    # Scale factor based on graph size and accuracy parameter
    scale = np.ceil(np.log2(n)) / epsilon

    # Setup preconditioner if requested
    if precon:
        M_inverse = sparse.linalg.spilu(laplacian)
        M = sparse.linalg.LinearOperator((n, n), M_inverse.solve)
    else:
        M = None

    if method == 'ext':  # Exact method - solve for each edge separately
        effective_resistance = np.zeros(shape=(1, m))
        for i in prange(m):
            # Extract the row corresponding to edge i
            Br = incidence_matrix[i, :].toarray()
            # Solve Laplacian system
            Z = cg(laplacian, Br.transpose(), tol=tol, M=M)[0]
            # Calculate effective resistance
            R_eff = Br @ Z
            effective_resistance[:, i] = R_eff[0]

        return effective_resistance[0]

    if method == 'spl':  # Splicing method - use random projections
        # Generate random projection matrices
        Q1 = sparse.random(int(scale), m, 1, format='csr') > 0.5
        Q2 = sparse.random(int(scale), m, 1, format='csr') > 0
        Q_not = Q1 - Q2
        Q = Q1 + (-1 * Q_not)  # Convert to {-1, 1} matrix
        Q = Q / np.sqrt(scale)  # Normalize

        # Create projected system
        SYS = Q @ weight_matrix @ incidence_matrix
        Z = np.zeros(shape=(int(scale), n))

        # Solve for each projection
        for i in prange(int(scale)):
            SYSr = SYS[i, :].toarray()
            Z[i, :] = cg(laplacian, SYSr.transpose(), tol=tol, M=M)[0]

        # Calculate effective resistance using projection results
        effective_resistance = np.sum(np.square(Z[:, edge_list[:, 0]] - Z[:, edge_list[:, 1]]), axis=0)
        return effective_resistance

    if method == 'kts':  # Koutis method
        effective_resistance_result = np.zeros(shape=(1, m))

        # Multiple random trials for approximation
        for i in prange(int(scale)):
            # Create random {-1, 1} vector
            ons1 = sparse.random(1, m, 1, format='csr') > 0.5
            ons2 = sparse.random(1, m, 1, format='csr') > 0
            ons_not = ons1 - ons2
            ons = ons1 + (-1 * ons_not)
            ons = ons / np.sqrt(scale)  # Normalize

            # Create and solve the system
            b = ons @ weight_matrix @ incidence_matrix
            b = b.toarray()
            Z = sparse.linalg.cg(laplacian, b.transpose(), tol=tol, M=M)[0]
            Z = Z.transpose()

            # Accumulate squared differences across edges
            effective_resistance_result = effective_resistance_result + np.abs(np.square(Z[edge_list[:, 0]] - Z[edge_list[:, 1]]))

        return effective_resistance_result[0]

def compute_angular_similarity(edge_index, node_features):
    """
    Compute the angular similarity for each edge based on node features.

    Args:
        edge_index: A tensor of shape (2, E) representing the edges
        node_features: A tensor of shape (n, f) representing node features

    Returns:
        torch.Tensor: A tensor of shape (E,) containing the angular similarities for each edge
    """
    # Extract features for both ends of each edge
    src_nodes = edge_index[0]
    dst_nodes = edge_index[1]
    src_features = node_features[src_nodes]
    dst_features = node_features[dst_nodes]

    # Compute cosine similarity components
    dot_products = (src_features * dst_features).sum(dim=1)
    src_norms = torch.norm(src_features, dim=1)
    dst_norms = torch.norm(dst_features, dim=1)

    # Calculate cosine similarity with numerical stability
    cos_sim = dot_products / (src_norms * dst_norms + 1e-5)
    cos_sim = torch.clamp(cos_sim, -1.0, 1.0)  # Ensure values in [-1, 1]

    # Map cosine similarity to [0, 1] range
    angular_sims = (1 + cos_sim) / 2

    return angular_sims


def expected_distinct_count(probabilities, sample_count):
    """
    Calculate the expected number of distinct elements sampled with replacement.
    
    Args:
        probabilities: Probabilities for each element
        sample_count: Number of samples to draw
        
    Returns:
        Expected number of distinct elements
    """
    return torch.sum(1 - (1 - probabilities) ** sample_count)


def add_missing_edges_and_update_types(edge_index, sampled_edge_index, edge_types):
    """
    Add missing edges from edge_index to sampled_edge_index and update edge_types accordingly.

    Args:
        edge_index: Tensor of size (2, E) containing all edges
        sampled_edge_index: Tensor of size (2, E') containing sampled edges
        edge_types: Tensor of size (E') containing edge types for the sampled edges

    Returns:
        (updated_sampled_edge_index, updated_edge_types): Tuple containing updated edge index and types
    """
    device = edge_index.device

    # Convert edge lists to sets for easier comparison
    edge_set = set(map(tuple, edge_index.T.tolist()))
    sampled_set = set(map(tuple, sampled_edge_index.T.tolist()))

    # Find edges that are in the original graph but not in the sampled graph
    missing_edges = edge_set - sampled_set
    missing_edges_tensor = torch.tensor(list(missing_edges), dtype=torch.long, device=device).T

    # Add missing edges to the sampled edges
    updated_sampled_edge_index = torch.cat((sampled_edge_index, missing_edges_tensor), dim=1)

    # Keep existing edge types and assign new type to missing edges
    updated_edge_types = edge_types.clone()
    E_prime = sampled_edge_index.shape[1]  # Original sampled edges count
    E_u = updated_sampled_edge_index.shape[1]  # Updated total count
    
    # Assign a new edge type (increment from max existing type)
    max_edge_type = torch.max(edge_types) if edge_types.numel() > 0 else -1
    new_edge_type = max_edge_type + 1
    new_edge_types_tensor = torch.full((E_u - E_prime,), new_edge_type, dtype=torch.long, device=device)

    # Combine existing and new edge types
    updated_edge_types = torch.cat((updated_edge_types, new_edge_types_tensor))

    return updated_sampled_edge_index, updated_edge_types

def find_required_samples(probabilities, target_distinct_count, max_iterations=100, tolerance=0.05):
    """
    Find the number of samples required to expect a certain number of distinct elements.
    
    Args:
        probabilities: Probabilities for each element
        target_distinct_count: Target number of distinct elements
        max_iterations: Maximum number of binary search iterations
        tolerance: Acceptable tolerance relative to target_distinct_count
        
    Returns:
        Required number of samples
    """
    num_edge = probabilities.shape[0]
    # Initialize binary search boundaries
    low, high = 1, 5*num_edge
    tolerance_threshold = tolerance * target_distinct_count
    iterations = 0

    # Binary search to find required sample count
    while iterations < max_iterations:
        sample_count = (low + high) // 2
        expected_count = expected_distinct_count(probabilities, sample_count)

        if abs(expected_count - target_distinct_count) < tolerance_threshold:
            return sample_count  # Found approximate solution
        elif expected_count < target_distinct_count:
            low = sample_count + 1  # Need more samples
        else:
            high = sample_count - 1  # Need fewer samples

        iterations += 1

    return high  # Return best estimate if max iterations reached

def sparsification(original_edge, edge_list, edge_weights, features, num_samples, num_relations=1, method='kts',
                   epsilon=0.1, device='cuda:0', undirected=True, keep_removed_edges=False, beta=1):
    """
    Samples edges from a graph based on angular similarity and effective resistance.

    Args:
        original_edge: Original edge index
        edge_list: numpy array of shape (E, 2) representing edges
        edge_weights: numpy array of shape (E,) representing weights of edges
        features: torch tensor of shape (n, f) representing node features
        num_samples: number of edges to sample
        num_relations: number of different edge types/relations to use
        method: method for effective resistance calculation ('ext', 'spl', 'kts')
        epsilon: parameter for effective resistance calculation
        device: device to use ('cpu' or 'cuda:0')
        undirected: if set to true, convert the sampled graph to an undirected graph
        keep_removed_edges: if set to true, keep removed edges as another type of relation
        beta: multiplier for the number of samples

    Returns:
        tuple: (sampled_edge_index, edge_type, sampled_edge_weight)
    """
    # Convert edge list to PyTorch tensor format
    edge_index = torch.LongTensor(edge_list.T).to(device)

    # Calculate edge importance measures
    angular_similarity = compute_angular_similarity(edge_index, features.to(device))
    effective_resistance = torch.tensor(compute_effective_resistance(edge_list, edge_weights, epsilon, method), device=device)

    # Combine importance measures to get sampling probabilities
    probabilities = (1 + 0.5*angular_similarity) * effective_resistance * torch.tensor(edge_weights, device=device)
    unnormalized_probabilities = probabilities.clone()
    probabilities /= torch.sum(probabilities)  # Normalize to sum to 1

    # Determine sample count needed to achieve target distinct edges
    sampling_count = find_required_samples(probabilities, int(num_samples*beta))

    # Calculate weights for inverse probability sampling
    inverse_probabilities = torch.tensor(edge_weights).to(device) / (sampling_count * probabilities)

    # Sample edges based on importance
    sampler = WeightedRandomSampler(unnormalized_probabilities, num_samples=sampling_count, replacement=True)
    sampled_indices = torch.LongTensor(list(sampler)).to(device)

    # Compute weighted edge indices
    sampled_weighted_edges = scatter_add_raw(inverse_probabilities[sampled_indices], sampled_indices, dim_size=edge_list.shape[0])

    # Create the sampled graph
    sampled_mask = sampled_weighted_edges > 0
    sampled_edge_index = edge_index[:, sampled_mask]
    sampled_edge_weight = torch.sqrt(sampled_weighted_edges[sampled_mask].float())

    print(f'# of edges of computational graph: {sampled_edge_index.shape[1]}')

    # Ensure undirected structure if requested
    if undirected:
        sampled_edge_index, sampled_edge_weight = to_undirected(sampled_edge_index, sampled_edge_weight, reduce='mean')

    # Convert edge weights to discrete types
    edge_types = (sampled_edge_weight.floor()).long()

    return sampled_edge_index, edge_types, sampled_edge_weight