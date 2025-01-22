import torch
from collections import defaultdict


def compute_pagerank(graph, num_nodes, alpha=0.85, max_iter=100, tol=1e-6, device='cuda'):
    """
    Compute the PageRank scores of nodes in the graph.

    Args:
        graph (defaultdict): Adjacency list representation of the graph.
        num_nodes (int): Total number of nodes in the graph.
        alpha (float): Damping factor for PageRank.
        max_iter (int): Maximum number of iterations.
        tol (float): Tolerance for convergence.

    Returns:
        torch.Tensor: A tensor of PageRank scores for each node.
    """
    # Initialize PageRank scores
    pagerank = torch.ones(num_nodes, dtype=torch.float32, device=device) / num_nodes
    dangling_weights = pagerank.clone()  # For handling dangling nodes
    for _ in range(max_iter):
        prev_pagerank = pagerank.clone()
        # Update PageRank scores
        for node in range(num_nodes):
            neighbors = graph[node]
            if len(neighbors) > 0:
                pagerank[node] = (1 - alpha) / num_nodes + alpha * sum(
                    prev_pagerank[n] / len(graph[n]) for n in neighbors)
            else:
                # Handle dangling nodes
                pagerank[node] = (1 - alpha) / num_nodes + alpha * sum(dangling_weights) / num_nodes

        # Check for convergence
        if torch.norm(pagerank - prev_pagerank, p=1) < tol:
            break

    return pagerank


def add_edges_based_on_lowest_degree_pagerank(edge_index, num_edges_to_add, num_nodes, device='cuda'):
    """
    Add `num_edges_to_add` edges to `edge_index` based on the minimum PageRank score sum of nodes
    with the lowest degrees.

    Args:
        edge_index (torch.Tensor): The existing edge list (2, E).
        num_edges_to_add (int): The number of edges to add.
        num_nodes (int): Total number of nodes in the graph (including isolated ones).
        device (str): The device to run the computation on ('cpu' or 'cuda').

    Returns:
        torch.Tensor: The updated edge list.
    """
    edge_index = edge_index.to(device)
    edge_list = edge_index.T.cpu().numpy()  # (E, 2)

    # Step 1: Create an adjacency list
    graph = defaultdict(list)
    for u, v in edge_list:
        graph[u].append(v)
        graph[v].append(u)  # Undirected graph

    # Step 2: Calculate degrees and include isolated nodes with degree 0
    degrees = {node: len(neighbors) for node, neighbors in graph.items()}

    # Add isolated nodes with degree 0 to the degree dictionary
    for i in range(num_nodes):
        if i not in degrees:
            degrees[i] = 0

    # Get the bottom nodes by degree
    num_candidates = min(num_nodes, 3 * num_edges_to_add)
    bottom_nodes_by_degree = sorted(degrees, key=degrees.get)[:num_candidates]

    # Step 3: Compute PageRank for the selected bottom nodes
    pagerank_tensor = compute_pagerank(graph, num_nodes, device=device).to(device)

    # Step 4: Sort selected nodes by PageRank score
    sorted_pagerank, sorted_idx = torch.sort(pagerank_tensor[bottom_nodes_by_degree])

    # Map sorted_idx back to the original node indices
    node_idx = [bottom_nodes_by_degree[i] for i in sorted_idx.tolist()]

    # Determine x such that x * (x - 1) / 2 >= num_edges_to_add
    x = 2  # Start with two nodes
    while x * (x - 1) // 2 < num_edges_to_add and x <= num_candidates:
        x += 1

    # Keep the first x elements of sorted_pagerank and node_idx
    sorted_pagerank = sorted_pagerank[:x]

    node_idx = node_idx[:x]


    # Step 5: Generate all possible pairs of the first x nodes
    potential_edges = [(node_idx[i], node_idx[j]) for i in range(x) for j in range(i + 1, x)]

    # Step 6: Compute the importance for each pair as P(u) + P(v)
    edge_scores = []
    for u, v in potential_edges:
        score = sorted_pagerank[node_idx.index(u)] + sorted_pagerank[node_idx.index(v)]
        edge_scores.append((score, u, v))

    # Step 7: Sort the edge scores and select the top `num_edges_to_add`
    edge_scores.sort(key=lambda x: x[0])  # Sort by score (P(u) + P(v))

    # Extract only the (u, v) pairs from the sorted edge_scores
    selected_edges = [(u, v) for _, u, v in edge_scores[:num_edges_to_add]]

    # Step 8: Convert selected edges to a tensor and add to edge_index
    new_edges = torch.tensor(selected_edges, dtype=torch.long, device=device).T  # Shape (2, num_edges_to_add)
    updated_edge_index = torch.cat((edge_index, new_edges), dim=1)

    return updated_edge_index


# import random
#
# def generate_complex_graph(num_nodes, num_edges):
#     """
#     Generate a complex graph with the specified number of nodes and edges.
#     Some nodes will be isolated.
#
#     Args:
#         num_nodes (int): Total number of nodes.
#         num_edges (int): Number of edges to create.
#
#     Returns:
#         torch.Tensor: Edge list as a tensor of shape (2, E).
#     """
#     edges = set()
#     while len(edges) < num_edges:
#         u = random.randint(0, num_nodes - 1)
#         v = random.randint(0, num_nodes - 1)
#         if u != v:
#             edges.add((u, v))
#
#     # Convert to edge index tensor
#     edge_index = torch.tensor(list(edges), dtype=torch.long).T  # Shape (2, E)
#     return edge_index
#
# # Parameters for the complex test
# num_nodes = 100  # Total number of nodes (including isolated ones)
# num_edges = 150  # Total number of edges in the graph
# num_edges_to_add = 10  # Number of edges to add based on PageRank
#
# # Generate a complex graph
# edge_index = generate_complex_graph(num_nodes, num_edges)
#
# # Print the initial edge_index
# print("Initial edge_index:\n", edge_index)
#
# # Use the previously defined function to compute the updated edge_index
# updated_edge_index = add_edges_based_on_lowest_degree_pagerank(edge_index, num_edges_to_add, num_nodes, device='cuda')
#
# # Print the updated edge_index
# print("Updated edge_index after adding edges based on PageRank:\n", updated_edge_index)