from collections import defaultdict, deque
import torch
import itertools
from torch_geometric.utils import degree, to_undirected, add_self_loops


def compute_laplacian_eigen(edge_index, num_nodes, max_iter=100, tol=1e-3, device="cuda"):
    """
    Computes the largest eigenvalue and leading eigenvector of the graph Laplacian
    using power iteration, ensuring all tensors are on the same device.

    Args:
        edge_index (torch.Tensor): Edge index of size (2, E) representing the edges.
        num_nodes (int): Number of nodes in the graph.
        max_iter (int): Maximum number of iterations for power iteration.
        tol (float): Convergence tolerance for power iteration.
        device (str): Device to use ('cpu' or 'cuda').

    Returns:
        float: Largest eigenvalue of the Laplacian.
        torch.Tensor: Leading eigenvector of size (num_nodes).
    """
    # Ensure edge_index is on the correct device
    edge_index = edge_index.to(device)

    # Step 1: Construct the Laplacian matrix
    row, col = edge_index
    edge_weights = torch.ones(row.size(0), device=device)  # Assume unweighted graph
    degree = torch.bincount(row, weights=edge_weights, minlength=num_nodes).to(device)

    # Create Laplacian components
    degree_matrix = torch.diag(degree)
    adjacency_matrix = torch.zeros((num_nodes, num_nodes), device=device)
    adjacency_matrix[row, col] = edge_weights
    laplacian_matrix = degree_matrix - adjacency_matrix

    # Step 2: Initialize a random vector for power iteration
    b_k = torch.rand(num_nodes, device=device)
    b_k = b_k / torch.norm(b_k)  # Normalize initial vector

    eigenvalue = 0.0

    for _ in range(max_iter):
        # Multiply by Laplacian matrix
        b_k1 = torch.matmul(laplacian_matrix, b_k)

        # Compute the Rayleigh quotient (approximation of eigenvalue)
        new_eigenvalue = torch.dot(b_k, b_k1)

        # Normalize the resulting vector
        b_k1_norm = torch.norm(b_k1)
        b_k = b_k1 / b_k1_norm

        # Check convergence
        if torch.abs(new_eigenvalue - eigenvalue) < tol:
            break

        eigenvalue = new_eigenvalue

    return eigenvalue.item(), b_k


def compute_k(edge_index, v_max, lambda_max, epsilon, degrees, device="cpu"):
    """
    Computes the number of edges (k) required to preserve the largest eigenvalue
    within an approximation error epsilon, using a refined method.

    Args:
        edge_index (torch.Tensor): Edge index of size (2, E) representing the edges.
        v_max (torch.Tensor): Leading eigenvector of the graph Laplacian (size |V|).
        lambda_max (float): Largest eigenvalue of the Laplacian.
        epsilon (float): Allowed approximation error for the largest eigenvalue.
        degrees (torch.Tensor): Degree of each node (size |V|).
        device (str): Device to use ('cpu' or 'cuda').

    Returns:
        int: The computed k value.
    """
    # Ensure tensors are on the correct device
    edge_index = edge_index.to(device)
    v_max = v_max.to(device)
    degrees = degrees.to(device)

    # Extract node pairs (u, v) from edge_index
    u, v = edge_index[0], edge_index[1]

    # Compute (f_u - f_v)^2 for all edges
    f_u = v_max[u]
    f_v = v_max[v]
    diff_squared = (f_u - f_v) ** 2

    # Compute weighted contributions w_uv = (f_u - f_v)^2 / (deg(u) + deg(v))
    deg_u = degrees[u]
    deg_v = degrees[v]
    weights = diff_squared / (deg_u + deg_v + 1e-6)  # Avoid division by zero

    # Compute the weighted average
    weighted_avg = weights.mean().item()

    # Compute k using the refined formula
    k = int(epsilon * lambda_max / weighted_avg)

    return k


def compute_fiedler_vector(edge_index, num_nodes, num_iterations=1000, tol=1e-5):
    # Add self-loops to avoid zero degrees
    edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)

    # Compute degrees
    row, col = edge_index
    degrees = degree(row, num_nodes, dtype=torch.float32)

    # Compute D^{-1/2} A D^{-1/2} as a sparse matrix
    inv_sqrt_deg = torch.pow(degrees, -0.5)
    inv_sqrt_deg[torch.isinf(inv_sqrt_deg)] = 0  # Handle zero degrees
    values = inv_sqrt_deg[row] * inv_sqrt_deg[col]

    # Create the normalized adjacency matrix as a sparse matrix
    normalized_adj = torch.sparse_coo_tensor(edge_index, values, (num_nodes, num_nodes)).to(edge_index.device)

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


def local_clustering(edge_index, subset_nodes):
    """
    Compute the local clustering coefficients for a subset of nodes and return them as a tensor.

    Args:
        edge_index (torch.Tensor): Tensor of size (2, E) representing edges of the graph (on GPU).
        subset_nodes (list): List of node indices for which to compute clustering coefficients.

    Returns:
        torch.Tensor: Tensor of clustering coefficients (one value per node in the subset).
    """
    # Step 1: Build adjacency list (neighbors of each node)
    num_nodes = int(edge_index.max().item()) + 1  # Total number of nodes
    adjacency_list = [set() for _ in range(num_nodes)]

    for src, dst in zip(edge_index[0].tolist(), edge_index[1].tolist()):
        adjacency_list[src].add(dst)
        adjacency_list[dst].add(src)  # Undirected graph

    # Step 2: Compute clustering coefficient for each node in the subset
    clustering_coeffs = []
    for node in subset_nodes:
        neighbors = adjacency_list[node]  # Neighbors of the node
        degree = len(neighbors)  # Degree of the node

        if degree < 2:
            # Clustering coefficient is 0 if there are less than 2 neighbors
            clustering_coeffs.append(0.0)
            continue

        # Count the number of edges between neighbors
        neighbor_edges = 0
        neighbors = list(neighbors)  # Convert to list for pairwise comparison
        for i in range(len(neighbors)):
            for j in range(i + 1, len(neighbors)):
                if neighbors[j] in adjacency_list[neighbors[i]]:
                    neighbor_edges += 1

        # Compute clustering coefficient
        possible_edges = degree * (degree - 1) / 2
        clustering_coeffs.append((2 * neighbor_edges) / possible_edges)

    # Convert clustering coefficients to a tensor
    return torch.tensor(clustering_coeffs, device=edge_index.device)


def bfs_closeness_centrality(graph, selected_nodes, num_nodes, device='cuda'):
    """
    Compute the closeness centrality for selected nodes using BFS for sparse graphs.

    Args:
        graph (defaultdict): Adjacency list representation of the graph.
        selected_nodes (list): List of nodes for which to compute closeness centrality.
        num_nodes (int): Total number of nodes in the graph.
        device (str): Device to run computations ('cuda' or 'cpu').

    Returns:
        torch.Tensor: A tensor of closeness centrality values for the selected nodes.
    """
    closeness_centrality = torch.zeros(len(selected_nodes), dtype=torch.float32, device=device)

    for idx, node in enumerate(selected_nodes):
        distances = torch.full((num_nodes,), float('inf'), device=device)
        distances[node] = 0  # Distance to itself is zero
        queue = deque([node])

        while queue:
            current = queue.popleft()
            current_distance = distances[current]

            # Traverse neighbors
            for neighbor in graph[current]:
                if distances[neighbor] == float('inf'):  # If not visited
                    distances[neighbor] = current_distance + 1
                    queue.append(neighbor)

        # Calculate closeness centrality for this node
        reachable_nodes = (distances < float('inf')).sum() - 1  # Exclude itself
        if reachable_nodes > 0:
            closeness_centrality[idx] = reachable_nodes / distances[distances < float('inf')].sum()

    return closeness_centrality


def compute_adjacency_list(edge_index, num_nodes):
    # Convert edge_index to adjacency list
    graph = defaultdict(list)
    edge_list = edge_index.t().tolist()
    for u, v in edge_list:
        graph[u].append(v)
        graph[v].append(u)  # Ensure undirected edges
    return graph




def edge_potential_score(u, v, fiedler_vector, closeness, degrees, selected_nodes_map):
    # Calculate Fiedler difference directly
    fiedler_diff = abs(fiedler_vector[u] - fiedler_vector[v])
    max_closeness = (closeness[selected_nodes_map[u]] + closeness[selected_nodes_map[v]])
    max_degree = (degrees[u] + degrees[v])
    # Compute the final score
    return fiedler_diff / (max_degree+1e-5)

def select_nodes_by_fiedler_and_degree(degrees, fiedler_vector, x):
    # Select 2x nodes with smallest and largest degrees
    smallest_degree_nodes = torch.topk(degrees, x, largest=False).indices
    largest_degree_nodes = torch.topk(degrees, x, largest=True).indices

    # Select 2x nodes with smallest and largest Fiedler vector components
    smallest_fiedler_nodes = torch.topk(fiedler_vector, x, largest=False).indices
    largest_fiedler_nodes = torch.topk(fiedler_vector, x, largest=True).indices

    # Combine all selected nodes and remove duplicates
    selected_nodes = torch.cat([largest_fiedler_nodes,
                                 smallest_fiedler_nodes]).unique()
    return selected_nodes


def select_edges_to_add(edge_index, num_nodes, k, beta=1, device='cuda'):
    # Step 1: Compute Fiedler vector and node degrees
    fiedler_vector = compute_fiedler_vector(edge_index, num_nodes)
    degrees = degree(edge_index[0], num_nodes, dtype=torch.float32)

    # test
    # lamb_max, v_max = compute_laplacian_eigen(edge_index, num_nodes)
    # k = compute_k(edge_index, v_max, lamb_max, epsilon=0.1, degrees=degrees)
    # k = min(k, 20)

    # Determine x such that x * (x - 1) / 2 >= num_edges_to_add
    x = 2  # Start with two nodes
    while x * (x - 1) // 2 < beta * k and x < num_nodes:
        x += 1

    # Step 2: Select nodes with smallest and largest Fiedler components and degrees (up to 4x nodes)
    selected_nodes = select_nodes_by_fiedler_and_degree(degrees, fiedler_vector, x)

    # Step 3: Compute closeness centrality for selected nodes
    # closeness = bfs_closeness_centrality(graph, selected_nodes.tolist(), num_nodes, device=device)
    closeness = local_clustering(edge_index, selected_nodes.tolist())

    # Map selected node indices to closeness indices
    selected_nodes_map = {node.item(): idx for idx, node in enumerate(selected_nodes)}


    # Step 5: Build candidate edges as all unique pairs of selected nodes
    candidate_edges = list(itertools.combinations(selected_nodes.tolist(), 2))

    # Step 6: Compute scores for all candidate edges
    edge_scores = [
        (edge_potential_score(u, v, fiedler_vector, closeness, degrees, selected_nodes_map),
         (u, v))
        for u, v in candidate_edges
    ]


    # Step 7: Select top k edges with the highest scores
    top_edges = sorted(edge_scores, key=lambda x: x[0], reverse=True)[:k]
    new_edges = [edge for score, edge in top_edges]

    # Step 8: Add selected edges to edge_index and return updated graph
    if new_edges:
        new_edge_index = torch.cat([edge_index, torch.tensor(new_edges, dtype=torch.long).t().to(edge_index.device)],
                                   dim=1)
    else:
        new_edge_index = edge_index
    return new_edge_index