from scipy.optimize import fsolve
import torch
import math


def add_edges_with_fiedler(edge_index, fiedler_vector, num_nodes, num_edges_to_add):
    """
    Add edges between nodes with the maximum |f_u - f_v| / (deg_u + deg_v + 1).
    Restricted to promising nodes for efficiency.

    Args:
        edge_index (torch.Tensor): Sparse edge index of shape (2, E).
        fiedler_vector (torch.Tensor): Fiedler vector of shape (num_nodes,).
        num_nodes (int): Total number of nodes in the graph.
        num_edges_to_add (int): Number of edges to add.

    Returns:
        torch.Tensor: Updated edge_index with added edges.
    """
    # Compute the degree of each node
    degrees = compute_node_degrees(edge_index, num_nodes)

    # Find the smallest d such that C_d^2 = d * (d - 1) / 2 >= 2 * num_edges_to_add
    d = math.ceil(0.5 * (1 + math.sqrt(1 + 16 * num_edges_to_add)))

    # Select promising nodes using two strategies
    # Get nodes with smallest degrees
    sorted_deg, deg_nodes = torch.sort(degrees)
    promising_deg_nodes = deg_nodes[:d]

    # Get nodes with largest absolute Fiedler values
    sorted_f, f_nodes = torch.sort(torch.abs(fiedler_vector), descending=True)
    promising_f_nodes = f_nodes[:d]

    # Combine both sets of promising nodes
    promising_nodes = torch.unique(torch.cat([promising_deg_nodes, promising_f_nodes]))

    # Filter out existing edges between promising nodes
    existing_edges = set(zip(edge_index[0].tolist(), edge_index[1].tolist()))
    promising_pairs = torch.combinations(promising_nodes, r=2)
    mask = torch.tensor([(u.item(), v.item()) not in existing_edges for u, v in promising_pairs],
                       device=edge_index.device)

    # Keep only promising pairs that are not already connected
    promising_pairs = promising_pairs[mask]

    # Iteratively add edges
    for _ in range(num_edges_to_add):
        # Compute scores for edges between all promising nodes
        u, v = promising_pairs[:, 0], promising_pairs[:, 1]
        score = torch.abs(fiedler_vector[u] - fiedler_vector[v])/(degrees[u] + degrees[v] + 1)

        # Find the pair with the maximum score
        max_idx = torch.argmax(score)
        node_u, node_v = u[max_idx], v[max_idx]

        # Add the edge (node_u, node_v) to edge_index
        new_edge = torch.tensor([[node_u], [node_v]], device=edge_index.device)
        edge_index = torch.cat([edge_index, new_edge], dim=1)

        # Update the degrees of node_u and node_v
        degrees[node_u] += 1
        degrees[node_v] += 1

        # Remove the selected pair from the list of promising pairs
        promising_pairs = torch.cat([promising_pairs[:max_idx], promising_pairs[max_idx + 1:]])

    return edge_index


def compute_node_degrees(edge_index, num_nodes, dtype=torch.float32):
    """
    Compute the degree of each node in the graph.
    
    Args:
        edge_index: Edge index tensor of shape (2, E)
        num_nodes: Number of nodes in the graph
        dtype: Data type for the output degrees tensor
        
    Returns:
        Tensor of node degrees
    """
    degrees = torch.zeros(num_nodes, dtype=dtype, device=edge_index.device)
    degrees.index_add_(0, edge_index[0], torch.ones(edge_index.shape[1], dtype=dtype, device=edge_index.device))
    degrees.index_add_(0, edge_index[1], torch.ones(edge_index.shape[1], dtype=dtype, device=edge_index.device))
    return degrees


def compute_fiedler_vector(edge_index, num_nodes, num_iterations=1000, tol=1e-5):
    """
    Approximate the Fiedler vector using power iteration on the normalized Laplacian.

    Args:
        edge_index (torch.Tensor): Sparse edge index of shape (2, E).
        num_nodes (int): Total number of nodes in the graph.
        num_iterations (int): Number of power iteration steps.
        tol (float): Convergence tolerance.

    Returns:
        torch.Tensor: Fiedler vector of shape (num_nodes,).
    """
    # Add self-loops to avoid zero degrees
    row, col = edge_index
    self_loops = torch.arange(num_nodes, device=edge_index.device).unsqueeze(0).repeat(2, 1)
    edge_index_with_loops = torch.cat([edge_index, self_loops], dim=1)

    # Compute degrees
    row, col = edge_index_with_loops
    degrees = torch.bincount(row, minlength=num_nodes).float()

    # Create normalized Laplacian: I - D^{-1/2} A D^{-1/2}
    inv_sqrt_deg = torch.pow(degrees, -0.5)
    inv_sqrt_deg[torch.isinf(inv_sqrt_deg)] = 0  # Handle zero degrees
    values = inv_sqrt_deg[row] * inv_sqrt_deg[col]

    # Create the normalized adjacency matrix as a sparse matrix
    normalized_adj = torch.sparse_coo_tensor(edge_index_with_loops, values, (num_nodes, num_nodes)).coalesce()

    # Power iteration to approximate Fiedler vector
    x = torch.rand(num_nodes, dtype=torch.float32, device=edge_index.device)
    for i in range(num_iterations):
        x_new = torch.sparse.mm(normalized_adj, x.view(-1, 1)).view(-1)
        x_new -= x_new.mean()  # Orthogonalize against constant vector
        x_new /= torch.norm(x_new)
        if torch.norm(x - x_new) < tol:
            break
        x = x_new
    return x


def equation_for_estimating_edges(n, m, q):
    """
    Equation to solve for estimating the number of edges in the latent graph.
    
    Args:
        n: Estimate for the true number of edges
        m: Observed number of edges
        q: Number of samples drawn
        
    Returns:
        Difference between expected number of distinct samples and m
    """
    return m - n * (1 - (1 - 1/n)**q)


def compute_sparse_laplacian_matrix(edge_index, num_nodes):
    """
    Compute the sparse graph Laplacian matrix L = D - A.
    
    Args:
        edge_index: Edge index tensor of shape (2, E)
        num_nodes: Number of nodes in the graph
        
    Returns:
        Sparse Laplacian matrix
    """
    # Extract source and target nodes
    row, col = edge_index
    
    # Calculate node degrees
    degrees = torch.zeros(num_nodes, dtype=torch.float32, device=edge_index.device).scatter_add_(
        0, row, torch.ones_like(row, dtype=torch.float32))

    # Create adjacency matrix in sparse format
    values = torch.ones(row.shape[0], device=edge_index.device)
    adjacency = torch.sparse_coo_tensor(edge_index, values, (num_nodes, num_nodes))

    # Create diagonal degree matrix
    degree_indices = torch.arange(num_nodes, device=edge_index.device).unsqueeze(0).repeat(2, 1)
    degree_matrix = torch.sparse_coo_tensor(degree_indices, degrees, (num_nodes, num_nodes))

    # Laplacian = D - A
    laplacian = degree_matrix - adjacency

    return laplacian


def power_iteration_sparse_matrix(laplacian, num_nodes, num_iter=100, tol=1e-3):
    """
    Power iteration method to compute the largest eigenvalue and eigenvector of a sparse matrix.
    
    Args:
        laplacian: Sparse Laplacian matrix
        num_nodes: Number of nodes in the graph
        num_iter: Maximum number of iterations
        tol: Convergence tolerance
        
    Returns:
        Tuple (eigenvalue, eigenvector)
    """
    # Initialize with random vector
    x = torch.rand(num_nodes, 1, device=laplacian.device)
    x = x / torch.norm(x)

    # Iteratively apply matrix multiplication and normalize
    for i in range(num_iter):
        x_new = torch.sparse.mm(laplacian, x)
        x_new = x_new / torch.norm(x_new)

        # Check convergence
        if torch.norm(x_new - x) < tol:
            break

        x = x_new

    # Calculate Rayleigh quotient for eigenvalue
    eigenvalue = (x.T @ torch.sparse.mm(laplacian, x)) / (x.T @ x)

    return eigenvalue.item(), x.squeeze()


def compute_max_edge_gradient(edge_index, eigenvector, eigenvalue):
    """
    Compute the maximum gradient of the eigenvector across edges, normalized by eigenvalue.
    
    Args:
        edge_index: Edge index tensor of shape (2, E)
        eigenvector: Eigenvector tensor
        eigenvalue: Corresponding eigenvalue
        
    Returns:
        Maximum normalized gradient value
    """
    u, v = edge_index[0], edge_index[1]  # Source and target nodes
    diff = eigenvector[u] - eigenvector[v]  # Eigenvector differences
    diff_squared = diff ** 2
    normalized_gradient = diff_squared / eigenvalue
    return torch.max(normalized_gradient)


def compute_average_degree(edge_index, num_nodes):
    """
    Compute the average degree of the graph.
    
    Args:
        edge_index: Edge index tensor of shape (2, E)
        num_nodes: Number of nodes in the graph
        
    Returns:
        Average degree (float)
    """
    degrees = compute_node_degrees(edge_index, num_nodes)
    return degrees.mean().item()


def update_sorted_position(sorted_degrees, node_ids, idx):
    """
    Update the position of a node in the sorted degree list after incrementing its degree.
    
    Args:
        sorted_degrees: Tensor of sorted node degrees
        node_ids: Tensor of node IDs corresponding to the sorted degrees
        idx: Index of the node whose degree was incremented
    """
    current_value = sorted_degrees[idx].item()
    insert_pos = idx
    
    # Find where the updated degree should be positioned
    while insert_pos + 1 < sorted_degrees.size(0) and sorted_degrees[insert_pos + 1] <= current_value:
        insert_pos += 1

    # Only reorder if position changes
    if insert_pos > idx:
        # Shift elements to make room
        sorted_degrees[idx:insert_pos] = sorted_degrees[idx + 1:insert_pos + 1].clone()
        sorted_degrees[insert_pos] = current_value

        # Update node IDs correspondingly
        node_ids[idx:insert_pos] = node_ids[idx + 1:insert_pos + 1].clone()
        node_ids[insert_pos] = node_ids[idx]


def add_edges_by_degrees(edge_index, num_nodes, num_edges_to_add):
    """
    Add edges by connecting nodes with the smallest degrees.
    
    Args:
        edge_index: Edge index tensor of shape (2, E)
        num_nodes: Number of nodes in the graph
        num_edges_to_add: Number of edges to add
        
    Returns:
        Updated edge_index tensor
    """
    # Get node degrees
    degrees = compute_node_degrees(edge_index, num_nodes, dtype=torch.int)
    
    # Sort nodes by degree
    sorted_degrees, node_ids = torch.sort(degrees)
    
    # Add edges between lowest-degree nodes
    for _ in range(num_edges_to_add):
        # Select the two nodes with lowest degrees
        node_u, node_v = node_ids[0], node_ids[1]
        
        # Add new edge
        new_edge = torch.tensor([[node_u], [node_v]], device=edge_index.device)
        edge_index = torch.cat([edge_index, new_edge], dim=1)
        
        # Update degrees
        degrees[node_u] += 1
        degrees[node_v] += 1
        
        # Update sorted array
        sorted_degrees[0] += 1
        sorted_degrees[1] += 1
        
        # Maintain order in sorted arrays
        update_sorted_position(sorted_degrees, node_ids, 0)
        update_sorted_position(sorted_degrees, node_ids, 1)
    
    return edge_index


def recover_latent_graph(edge_index, num_nodes, k_guess, step_size, metric='degree', advanced=True, beta=1.0):
    """
    Recover a latent graph structure by adding edges to the observed graph.
    
    Args:
        edge_index: Observed edge index tensor of shape (2, E)
        num_nodes: Number of nodes in the graph
        k_guess: Initial guess for the number of edges to add
        step_size: Step size for increasing epsilon in the search
        metric: Method for selecting edges to add ('degree' or other)
        advanced: Whether to use advanced method with Fiedler vector
        beta: Scaling factor for calculations
        
    Returns:
        Enhanced edge_index tensor
    """
    print(f'# of edges of input graph: {edge_index.shape[1]}')
    
    # Construct graph Laplacian
    laplacian = compute_sparse_laplacian_matrix(edge_index, num_nodes)

    # Get spectral information through power iteration
    eigenvalue, eigenvector = power_iteration_sparse_matrix(laplacian, num_nodes)

    # Measure maximum eigenvector gradient across edges
    max_gradient = compute_max_edge_gradient(edge_index, eigenvector, eigenvalue)

    # Get structural information
    avg_degree = compute_average_degree(edge_index, num_nodes)

    # Setup parameters for edge estimation
    m = edge_index.shape[1]  # Current edge count
    initial_guess = m * 1.5
    max_epsilon = 2
    epsilon = step_size
    log_8 = math.log(8)

    # Formula from theory: approximation quality parameter
    final_value = ((max_gradient * m / avg_degree) ** 2) / 2 * log_8

    # Search for appropriate number of edges to add
    while epsilon <= max_epsilon:
        # Calculate sample requirements based on epsilon
        q = int(final_value / (epsilon ** 2))
        if q <= m:
            return edge_index

        # Solve for estimated true edge count
        n_estimate = fsolve(equation_for_estimating_edges, initial_guess, args=(m, q))[0]

        # Check if estimate exceeds threshold
        if int(n_estimate) >= m + k_guess:
            break
            
        # Increase approximation parameter
        step_size = min(step_size, 0.01)
        epsilon += step_size
        
    # Calculate edges to add
    num_edges_to_add = int(n_estimate - m)

    # Add edges based on chosen strategy
    if metric == 'degree':
        if advanced:
            # Use spectral properties to guide edge addition
            fiedler_vector = compute_fiedler_vector(edge_index, num_nodes)
            edge_index = add_edges_with_fiedler(edge_index, fiedler_vector, num_nodes, num_edges_to_add)
        else:
            # Simple strategy: connect lowest-degree nodes
            edge_index = add_edges_by_degrees(edge_index, num_nodes, num_edges_to_add)
                
    print(f'# of edges of latent graph: {edge_index.shape[1]}')
    return edge_index


