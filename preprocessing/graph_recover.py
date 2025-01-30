from scipy.optimize import fsolve
import torch
import math


def add_edges(edge_index, f, num_nodes, num_edges_to_add):
    """
    Add edges between nodes with the maximum |f_u - f_v| / (deg_u + deg_v + 1).
    Restricted to promising nodes for efficiency.

    Args:
        edge_index (torch.Tensor): Sparse edge index of shape (2, E).
        f (torch.Tensor): Fiedler vector of shape (num_nodes,).
        num_nodes (int): Total number of nodes in the graph.
        num_edges_to_add (int): Number of edges to add.

    Returns:
        torch.Tensor: Updated edge_index with added edges.
    """
    #Compute the degree of each node
    degrees = torch.zeros(num_nodes, dtype=torch.float32, device=edge_index.device)
    degrees.index_add_(0, edge_index[0], torch.ones(edge_index.shape[1], dtype=torch.float32, device=edge_index.device))
    degrees.index_add_(0, edge_index[1], torch.ones(edge_index.shape[1], dtype=torch.float32, device=edge_index.device))

    #Compute d
    # Find the smallest d such that C_d^2 = d * (d - 1) / 2 >= 2 * num_edges_to_add
    d = math.ceil(0.5 * (1 + math.sqrt(1 + 16 * num_edges_to_add)))

    #Identify promising nodes
    # 3.1 Nodes with smallest degrees
    sorted_deg, deg_nodes = torch.sort(degrees)  # Sort nodes by degrees
    promising_deg_nodes = deg_nodes[:d]  # Take d nodes with smallest degrees

    # Nodes with largest absolute f values
    sorted_f, f_nodes = torch.sort(torch.abs(f), descending=True)
    promising_f_nodes = f_nodes[:d]  # Take d nodes with largest |f|


    # # Union of both sets of promising nodes
    promising_nodes = torch.unique(torch.cat([promising_deg_nodes, promising_f_nodes]))

    # # Create a mask to filter out existing edges
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
        score = abs(f[u] - f[v])/(degrees[u] + degrees[v] + 1)

        # Find the pair with the maximum score
        max_idx = torch.argmax(score)
        node_u, node_v = u[max_idx], v[max_idx]

        # Add the edge (node_u, node_v) to edge_index
        new_edge = torch.tensor([[node_u], [node_v]], device=edge_index.device)

        edge_index = torch.cat([edge_index, new_edge], dim=1)

        # Update the degrees of node_u and node_v
        degrees[node_u] += 1
        degrees[node_v] += 1

        # Update the magnitudes of f[node_u] and f[node_v]
        f[node_u] *= (1 - 1 / num_edges_to_add)
        f[node_v] *= (1 - 1 / num_edges_to_add)

        # Remove the selected pair and its score from the lists
        promising_pairs = torch.cat([promising_pairs[:max_idx], promising_pairs[max_idx + 1:]])

    return edge_index


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
    edge_index = torch.cat([edge_index, self_loops], dim=1)

    # Compute degrees
    row, col = edge_index
    degrees = torch.bincount(row, minlength=num_nodes).float()

    # Compute D^{-1/2} A D^{-1/2} as a sparse matrix
    inv_sqrt_deg = torch.pow(degrees, -0.5)
    inv_sqrt_deg[torch.isinf(inv_sqrt_deg)] = 0  # Handle zero degrees
    values = inv_sqrt_deg[row] * inv_sqrt_deg[col]

    # Create the normalized adjacency matrix as a sparse matrix
    normalized_adj = torch.sparse_coo_tensor(edge_index, values, (num_nodes, num_nodes)).coalesce()

    # Power iteration to approximate Fiedler vector
    x = torch.rand(num_nodes, dtype=torch.float32, device=edge_index.device)
    for _ in range(num_iterations):
        x_new = torch.sparse.mm(normalized_adj, x.view(-1, 1)).view(-1)
        x_new -= x_new.mean()  # Ensure orthogonality with the constant vector
        x_new /= torch.norm(x_new)
        if torch.norm(x - x_new) < tol:
            break
        x = x_new
    return x

def equation_for_n(n, m, q):
    return m - n * (1 - (1 - 1/n)**q)

# Function to compute the sparse graph Laplacian
def compute_sparse_laplacian(edge_index, num_nodes):
    # Degree matrix (D)
    row, col = edge_index
    deg = torch.zeros(num_nodes, dtype=torch.float32, device=edge_index.device).scatter_add_(0, row, torch.ones_like(row, dtype=torch.float32))

    # Create adjacency matrix (A) in sparse format
    values = torch.ones(row.shape[0], device=edge_index.device)
    A = torch.sparse_coo_tensor(edge_index, values, (num_nodes, num_nodes))

    # Create sparse degree matrix D
    D_indices = torch.arange(num_nodes, device=edge_index.device).unsqueeze(0).repeat(2, 1)  # Diagonal indices for D
    D = torch.sparse_coo_tensor(D_indices, deg, (num_nodes, num_nodes))

    # Compute the sparse Laplacian matrix L = D - A
    L = D - A

    return L

# Power Iteration to compute the largest eigenvalue and eigenvector using sparse matrix multiplication
def power_iteration_sparse(L, num_nodes, num_iter=100, tol=1e-3):
    # Random initial vector
    x = torch.rand(num_nodes, 1, device=L.device)
    x = x / torch.norm(x)  # Normalize initial vector

    # Power iteration
    for _ in range(num_iter):
        # Sparse-dense matrix multiplication: L @ x
        x_new = torch.sparse.mm(L, x)

        # Normalize the new vector
        x_new = x_new / torch.norm(x_new)

        # Check for convergence
        if torch.norm(x_new - x) < tol:
            break

        x = x_new

    # Rayleigh quotient to approximate the largest eigenvalue
    sigma = (x.T @ torch.sparse.mm(L, x)) / (x.T @ x)

    return sigma.item(), x.squeeze()

# Function to compute the maximum value of (x_u - x_v)^2 / sigma
def compute_max_edge_value(edge_index, x, sigma):
    u = edge_index[0]  # Source nodes
    v = edge_index[1]  # Target nodes

    # Compute the differences x_u - x_v
    diff = x[u] - x[v]

    # Square the differences
    diff_squared = diff ** 2

    # Divide by sigma
    result = diff_squared / sigma

    # Return the maximum value
    max_value = torch.max(result)

    return max_value

# Function to compute the average degree of the graph
def compute_avg_degree(edge_index, num_nodes):
    row, col = edge_index
    deg = torch.zeros(num_nodes, dtype=torch.float32, device=edge_index.device).scatter_add_(0, row, torch.ones_like(row, dtype=torch.float32))
    avg_degree = (deg.mean()).item()
    return avg_degree


def update_position(sorted_deg, node_id, idx):
    """Find the correct position for sorted_deg[idx] in the sorted list."""
    current_value = sorted_deg[idx].item()
    insert_pos = idx
    # Find the correct position for sorted_deg[idx]
    while insert_pos + 1 < sorted_deg.size(0) and sorted_deg[insert_pos + 1] <= current_value:
        insert_pos += 1

    # Move the value to the correct position (after elements greater than it)
    if insert_pos > idx:
        sorted_deg[idx:insert_pos] = sorted_deg[idx + 1:insert_pos + 1].clone()
        sorted_deg[insert_pos] = current_value

        # Do the same for node_id
        node_id[idx:insert_pos] = node_id[idx + 1:insert_pos + 1].clone()
        node_id[insert_pos] = node_id[idx]

# Main function
def recover_latent_graph(edge_index, num_nodes, k_guess, step_size, metric='degree', advanced=True, beta=1.0):
    print(f'# of edges of input graph: {edge_index.shape[1]}')
    # Compute the Laplacian matrix as a sparse matrix
    L = compute_sparse_laplacian(edge_index, num_nodes)


    # Compute the largest eigenvalue and corresponding eigenvector using power iteration with sparse matrix
    sigma, x = power_iteration_sparse(L, num_nodes)

    # Compute the maximum value for the edges
    max_value = compute_max_edge_value(edge_index, x, sigma)

    # Compute the average degree
    avg_degree = compute_avg_degree(edge_index, num_nodes)

    # Compute the number of edges
    num_edges = edge_index.size(1)  # Size of the second dimension of edge_index

    # Compute the log(8)
    log_8 = math.log(8)

    m = edge_index.shape[1]
    initial_guess = m * 1.5
    max_epsilon = 2
    epsilon = step_size

    # Compute the final value: (max_value * #E / avg_degree)^2 / 2 / epsilon^2 * log(8)
    final_value = ((max_value * num_edges / avg_degree) ** 2) / (2) * log_8


    while epsilon <= max_epsilon:
        # Recalculate q with the updated epsilon
        q = int(final_value / (epsilon ** 2))
        if q<=m:
            return edge_index

        # Solve for n_estimate using fsolve
        n_estimate = fsolve(equation_for_n, initial_guess, args=(m, q))[0]

        # Check if n_estimate is larger than m
        if int(n_estimate) >= m + k_guess:
            # print(f"Estimated n = {int(n_estimate)} is larger than m = {m}")
            break
        # Increase epsilon by 0.01 for the next iteration
        step_size = min(step_size, 0.01)
        epsilon += step_size
    num_edges_to_add = int(n_estimate - m)  # Number of edges to add


    if metric == 'degree':
        if advanced == True:
            f = compute_fiedler_vector(edge_index, num_nodes)
            edge_index = add_edges(edge_index, f, num_nodes, num_edges_to_add)
        else:
            # Step 1: Compute the degree of each node
            degrees = torch.zeros(num_nodes, dtype=torch.int, device=edge_index.device)
            degrees.index_add_(0, edge_index[0], torch.ones(edge_index.shape[1], dtype=torch.int, device=edge_index.device))
            degrees.index_add_(0, edge_index[1], torch.ones(edge_index.shape[1], dtype=torch.int, device=edge_index.device))


            # Step 2: Initialize sorted_deg and node_id based on the current degrees
            sorted_deg, node_id = torch.sort(degrees)  # sorted_deg has sorted degrees, node_id has corresponding node IDs

            for _ in range(num_edges_to_add):
                # Find the nodes with the two smallest degrees
                node_u, node_v = node_id[0], node_id[1]  # Get the IDs of the nodes with the smallest degrees

                # Add the edge (node_u, node_v) to the edge_index
                new_edge = torch.tensor([[node_u], [node_v]], device=edge_index.device)
                edge_index = torch.cat([edge_index, new_edge], dim=1)

                # Update the degrees of node_u and node_v
                degrees[node_u] += 1
                degrees[node_v] += 1

                # Update sorted_deg[0] and sorted_deg[1] (we only need to adjust these two)
                sorted_deg[0] += 1  # Update the degree of node_u
                sorted_deg[1] += 1  # Update the degree of node_v

                # Now we need to find the correct position for sorted_deg[0] and sorted_deg[1]
                update_position(sorted_deg, node_id, 0)  # Correct position for updated node_u
                update_position(sorted_deg, node_id, 1)
    print(f'# of edges of latent graph: {edge_index.shape[1]}')
    return edge_index


