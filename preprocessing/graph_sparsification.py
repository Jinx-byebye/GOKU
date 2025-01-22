import numpy as np
from scipy import sparse
from scipy.sparse.linalg import cg
from numba import njit, prange
import torch
from torch.utils.data import WeightedRandomSampler
from torch_geometric.utils import to_undirected, undirected
from typing import Optional, Tuple


# torch_scatter/utils.py
def broadcast(src: torch.Tensor, other: torch.Tensor, dim: int):
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
    return scatter_sum_raw(src, index, dim, out, dim_size)


@njit(nopython=True)
def Mtrx_Elist(A):
    """
    Convert an adjacency matrix to an edge list.

    Args:
        A (ndarray): Adjacency matrix.

    Returns:
        tuple: Edge list and corresponding weights.
    """
    j, i = np.nonzero(np.triu(A))  # Identify non-zero edges in the upper triangle
    elist = np.vstack((i, j))
    weights = A[np.triu(A) != 0]  # Extract weights corresponding to edges

    return elist.transpose(), weights

@njit(nopython=True)
def Elist_Mtrx(E_list, weights):
    """
    Convert an edge list to an adjacency matrix.

    Args:
        E_list (ndarray): Edge list.
        weights (ndarray): Corresponding edge weights.

    Returns:
        ndarray: Adjacency matrix.
    """
    n = np.max(E_list) + 1  # Determine number of nodes
    A = np.zeros(shape=(n, n))

    for i in range(np.shape(E_list)[0]):
        n1, n2 = E_list[i, :]
        w = weights[i]
        A[n1, n2], A[n2, n1] = w, w  # Fill adjacency matrix symmetrically

    return A


def Elist_Mtrx_s(E_list, weights):
    """
    Convert an edge list to a sparse adjacency matrix.

    Args:
        E_list (ndarray): Edge list.
        weights (ndarray): Corresponding edge weights.

    Returns:
        csr_matrix: Sparse adjacency matrix in CSR format.
    """
    n = np.max(E_list) + 1  # Determine number of nodes
    A = sparse.csr_matrix((weights, (E_list[:, 0], E_list[:, 1])), shape=(n, n))
    A = A + A.transpose()  # Ensure the matrix is symmetric

    return A

@njit(nopython=True)
def Lap(A):
    """
    Compute the Laplacian matrix from an adjacency matrix.

    Args:
        A (ndarray): Adjacency matrix.

    Returns:
        ndarray: Laplacian matrix.
    """
    L = np.diag(np.sum(abs(A), 1)) - A  # Calculate Laplacian as D - A
    return L


def Lap_s(A):
    """
    Compute the Laplacian matrix from a sparse adjacency matrix.

    Args:
        A (csr_matrix): Sparse adjacency matrix.

    Returns:
        csr_matrix: Laplacian matrix in sparse format.
    """
    L = sparse.csgraph.laplacian(A)  # Directly compute Laplacian for sparse matrices
    return L


def sVIM(E_list):
    """
    Compute the signed-edge vertex incidence matrix.

    Args:
        E_list (ndarray): Edge list.

    Returns:
        csr_matrix: Sparse vertex incidence matrix.
    """
    m = np.shape(E_list)[0]  # Number of edges
    E_list = E_list.transpose()  # Transpose to make rows correspond to edges

    data = [1] * m + [-1] * m  # Create data for incidence matrix
    i = list(range(0, m)) + list(range(0, m))  # Row indices
    j = E_list[0, :].tolist() + E_list[1, :].tolist()  # Column indices

    B = sparse.csr_matrix((data, (i, j)))  # Create sparse matrix in CSR format

    return B


def WDiag(weights):
    """
    Compute the diagonal weights matrix.

    Args:
        weights (ndarray): Edge weights.

    Returns:
        dia_matrix: Diagonal matrix of square root of weights.
    """
    m = len(weights)
    weights_sqrt = np.sqrt(weights)  # Element-wise square root of weights
    W = sparse.dia_matrix((weights_sqrt, [0]), shape=(m, m))  # Diagonal sparse matrix

    return W


def compute_effective_resistance(E_list, weights, epsilon, type='kts', tol=1e-10, precon=False):
    """
    Approximate effective resistance using various methods.

    Args:
        E_list (ndarray): Edge list.
        weights (ndarray): List of edge weights.
        epsilon (float): Accuracy control parameter.
        type (str): Type of calculation ('ext', 'spl', 'kts').
        tol (float): Tolerance for convergence (default: 1e-10).
        precon (bool or sparse.LinearOperator): Preconditioner for the solver (default: False).

    Returns:
        ndarray: Effective resistance values.
    """
    m = np.shape(E_list)[0]  # Number of edges
    n = np.max(E_list) + 1  # Number of nodes

    A = Elist_Mtrx_s(E_list, weights)  # Get sparse adjacency matrix
    L = Lap_s(A)  # Compute sparse Laplacian
    B = sVIM(E_list)  # Compute vertex incidence matrix
    W = WDiag(weights)  # Compute diagonal weight matrix
    scale = np.ceil(np.log2(n)) / epsilon  # Calculate scale for projection

    if precon:
        M_inverse = sparse.linalg.spilu(L)  # Compute incomplete LU for preconditioning
        M = sparse.linalg.LinearOperator((n, n), M_inverse.solve)

    else:
        M = None  # No preconditioner

    if type == 'ext':
        effR = np.zeros(shape=(1, m))  # Initialize effective resistance array
        for i in prange(m):
            Br = B[i, :].toarray()  # Row vector for the i-th edge
            Z = cg(L, Br.transpose(), tol=tol, M=M)[0]  # Solve for Z
            R_eff = Br @ Z  # Calculate effective resistance
            effR[:, i] = R_eff[0]

        return effR[0]

    if type == 'spl':
        Q1 = sparse.random(int(scale), m, 1, format='csr') > 0.5  # Generate random matrix Q1
        Q2 = sparse.random(int(scale), m, 1, format='csr') > 0  # Generate random matrix Q2
        Q_not = Q1 - Q2  # Logical NOT for the random matrices
        Q = Q1 + (-1 * Q_not)  # Combine to form a matrix of 1s and -1s
        Q = Q / np.sqrt(scale)  # Normalize Q

        SYS = Q @ W @ B  # Create system for projection
        Z = np.zeros(shape=(int(scale), n))  # Initialize solution matrix

        for i in prange(int(scale)):
            SYSr = SYS[i, :].toarray()  # Row system for projection
            Z[i, :] = cg(L, SYSr.transpose(), tol=tol, M=M)[0]  # Solve for Z

        effR = np.sum(np.square(Z[:, E_list[:, 0]] - Z[:, E_list[:, 1]]), axis=0)  # Calculate effective resistance
        return effR

    if type == 'kts':
        effR_res = np.zeros(shape=(1, m))  # Initialize results

        for i in prange(int(scale)):
            ons1 = sparse.random(1, m, 1, format='csr') > 0.5  # Random row vector
            ons2 = sparse.random(1, m, 1, format='csr') > 0  # Another random row vector
            ons_not = ons1 - ons2  # Logical NOT operation
            ons = ons1 + (-1 * ons_not)  # Create matrix of 1s and -1s
            ons = ons / np.sqrt(scale)  # Normalize

            b = ons @ W @ B  # Compute b vector for current iteration
            b = b.toarray()  # Convert to dense format

            Z = sparse.linalg.cg(L, b.transpose(), tol=tol, M=M)[0]  # Solve for Z
            Z = Z.transpose()  # Transpose for effective resistance calculation

            effR_res = effR_res + np.abs(np.square(Z[E_list[:, 0]] - Z[E_list[:, 1]]))  # Accumulate results

        return effR_res[0]

def compute_angular_similarity(edge_index, node_features):
    """
    Compute the angular similarity for each edge based on node features.

    Args:
        edge_index (torch.Tensor): A tensor of shape (2, E) representing the edges.
        node_features (torch.Tensor): A tensor of shape (n, f) representing node features.

    Returns:
        torch.Tensor: A tensor of shape (E,) containing the angular similarities for each edge.
    """
    # Extract source and target node indices from edge_index
    src_nodes = edge_index[0]  # Indices of source nodes
    dst_nodes = edge_index[1]  # Indices of destination nodes

    # Gather the node features for source and destination nodes
    src_features = node_features[src_nodes]  # Shape: (E, f)
    dst_features = node_features[dst_nodes]  # Shape: (E, f)

    # Compute the dot product and norms
    dot_products = (src_features * dst_features).sum(dim=1)  # Shape: (E,)
    src_norms = torch.norm(src_features, dim=1)  # Shape: (E,)
    dst_norms = torch.norm(dst_features, dim=1)  # Shape: (E,)

    # Compute cosine similarity
    cos_sim = dot_products / (src_norms * dst_norms + 1e-5)

    # Clamp values to avoid numerical errors
    cos_sim = torch.clamp(cos_sim, -1.0, 1.0)

    # Compute angular similarity
    # angular_sims = 1 - (torch.acos(cos_sim) / torch.pi)
    angular_sims = (1 + cos_sim) / 2

    return angular_sims


def expected_distinct_count(P, s):
    """Calculate the expected number of distinct elements sampled."""
    return torch.sum(1 - (1 - P) ** s)


def add_missing_edges_and_update_types(edge_index, sampled_edge_index, edge_types):
    """
    Add missing edges from edge_index to sampled_edge_index and update edge_types accordingly.

    Args:
        edge_index (torch.Tensor): Tensor of size (2, E) containing all edges.
        sampled_edge_index (torch.Tensor): Tensor of size (2, E') containing sampled edges.
        edge_types (torch.Tensor): Tensor of size (E') containing edge types for the sampled edges.

    Returns:
        updated_sampled_edge_index (torch.Tensor): Updated sampled edge index.
        updated_edge_types (torch.Tensor): Updated edge types with new types assigned to the added edges.
    """
    # Ensure tensors are on the same device
    device = edge_index.device

    # Convert edge_index and sampled_edge_index to sets of edges
    edge_set = set(map(tuple, edge_index.T.tolist()))  # Convert edges to a set of tuples
    sampled_set = set(map(tuple, sampled_edge_index.T.tolist()))  # Convert sampled edges to a set of tuples

    # Find the missing edges (present in edge_index but not in sampled_edge_index)
    missing_edges = edge_set - sampled_set  # Edges present in edge_index but not in sampled_edge_index

    # Convert the missing edges back to a tensor
    missing_edges_tensor = torch.tensor(list(missing_edges), dtype=torch.long,
                                        device=device).T  # Transpose to (2, E_missing)

    # Concatenate the missing edges to the sampled_edge_index
    updated_sampled_edge_index = torch.cat((sampled_edge_index, missing_edges_tensor), dim=1)

    # Create the updated_edge_types
    updated_edge_types = edge_types.clone()  # Copy the original edge types

    # The first E' elements of updated_edge_types are kept the same
    E_prime = sampled_edge_index.shape[1]  # The original number of sampled edges (E')
    E_u = updated_sampled_edge_index.shape[1]  # The updated number of sampled edges (E_u)

    # For the added edges (from the missing edges), assign new edge types
    max_edge_type = torch.max(edge_types) if edge_types.numel() > 0 else -1  # max edge type, or -1 if empty
    new_edge_type = max_edge_type + 1  # New edge type to assign

    # Assign the new edge types to the newly added edges (those after E')
    new_edge_types_tensor = torch.full((E_u - E_prime,), new_edge_type, dtype=torch.long, device=device)

    # Concatenate the new edge types to the existing edge types
    updated_edge_types = torch.cat((updated_edge_types, new_edge_types_tensor))

    return updated_sampled_edge_index, updated_edge_types

def find_required_samples(P, m, max_iterations=100, tolerance=0.05):
    """Find the number of samples required to expect m distinct elements."""
    num_edge = P.shape[0]
    low, high = 1, 5*num_edge  # Define initial bounds for s
    iterations = 0
    tolerance = tolerance*m


    while iterations < max_iterations:
        s = (low + high) // 2  # Binary search midpoint
        expected_count = expected_distinct_count(P, s)

        if abs(expected_count - m) < tolerance:
            return s  # Found the required number of samples
        elif expected_count < m:
            low = s + 1  # Increase the lower bound
        else:
            high = s - 1  # Decrease the upper bound

        iterations += 1

    return high  # If not found within max_iterations

def sparsification(original_edge, edge_list, edge_weights, features, num_samples,  num_relations=1, method='kts',
                   epsilon=0.1, device='cuda:0', undirected=True, keep_removed_edges=False, beta=1):
    """
    Samples edges from a graph based on angular similarity and effective resistance.

    Parameters:
    - edge_list: numpy array of shape (E, 2) representing edges
    - edge_weights: numpy array of shape (E,) representing weights of edges
    - features: torch tensor of shape (n, f) representing node features
    - num_samples: number of edges to sample
    - method: method for effective resistance calculation ('ext', 'spl', 'kts')
    - epsilon: parameter for effective resistance calculation
    - device: device to use ('cpu' or 'cuda:0')
    - undirected: if set to true, convert the sampled graph to an undirected graph
    - keep_removed_edges: if set to true, keep removed edges as another type of relation

    Returns:
    - sampled_edge_index: tensor of shape (2, S) for sampled edges
    - edge_type: tensor of shape (S,) representing edge types
    - sampled_edge_weight: tensor of shape (S,) for weights of sampled edges
    - edge_probabilities: tensor of probabilities for each edge
    """
    edge_index = torch.LongTensor(edge_list.T).to(device)  # Convert edge_list to a PyTorch LongTensor

    # Compute angular similarity between edges and node features
    angular_similarity = compute_angular_similarity(edge_index, features.to(device))

    # Calculate effective resistance for each edge
    effective_resistance = torch.tensor(compute_effective_resistance(edge_list, edge_weights, epsilon, method), device=device)

    # Calculate probabilities for edge sampling
    probabilities = (1 + 0.5*angular_similarity) * effective_resistance * torch.tensor(edge_weights, device=device)

    unnormalized_probabilities = probabilities
    probabilities /= torch.sum(probabilities)  # Normalize probabilities to sum to 1


    # compute sampling count q
    q = find_required_samples(probabilities, int(num_samples*beta))

    # Calculate the inverse probability for weighted sampling
    inverse_probabilities = torch.tensor(edge_weights).to(device) / (q * probabilities)


    # Use WeightedRandomSampler for sampling edges

    sampler = WeightedRandomSampler(unnormalized_probabilities, num_samples=q, replacement=True)
    sampled_indices = torch.LongTensor(list(sampler)).to(device)


    # Compute weighted edge indices using scatter_add
    sampled_weighted_edges = scatter_add_raw(inverse_probabilities[sampled_indices], sampled_indices, dim_size=edge_list.shape[0])

    # Create a boolean mask to identify sampled edges
    sampled_mask = sampled_weighted_edges > 0

    # Sample the edge indices and weights
    sampled_edge_index = edge_index[:, sampled_mask]
    sampled_edge_weight = torch.sqrt(sampled_weighted_edges[sampled_mask].float())


    print(f'# of edges of computational graph: {sampled_edge_index.shape[1]}')

    if undirected:
        # sampled_edge_index = to_undirected(sampled_edge_index)
        sampled_edge_index, sampled_edge_weight = to_undirected(sampled_edge_index, sampled_edge_weight, reduce='mean')

    # create the categories based on integer values
    edge_types = (sampled_edge_weight.floor()).long()  # floor and cast to integer

    return sampled_edge_index, edge_types, sampled_edge_weight