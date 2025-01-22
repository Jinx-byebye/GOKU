import torch


def laser(edge_index, p, max_k):
    #Initialize adjacency matrix A from edge_index
    num_nodes = edge_index.max().item() + 1  # assuming nodes are indexed from 0 to max node id
    A = torch.zeros((num_nodes, num_nodes), device=edge_index.device)
    A = A + torch.eye(A.size(0), device=A.device)

    # Populate adjacency matrix A based on edge_index
    A[edge_index[0], edge_index[1]] = 1

    # Step 2: Initialize variables for power computations
    A_prev = A.clone()  # A^1 = A (initial adjacency matrix)
    updates_to_set = []  # List to store updates to adjacency matrix

    #Compute powers of A iteratively
    for t in range(2, max_k + 1):
        # Compute A^t as A^(t-1) * A
        A_current = torch.matmul(A_prev, A)  # A^t

        # Identify new edges: A^t_ij > 0 and A^(t-1)_ij == 0
        new_edges = (A_current > 0) & (A_prev == 0)

        # Step 4: Record the p-fraction of smallest values for each node
        for i in range(num_nodes):
            # Get the indices of new edges for node i
            indices = (new_edges[i] > 0).nonzero(as_tuple=True)[0]
            if len(indices) > 0:
                # Get the values of A^t_ij for the new edges
                edge_values = A_current[i, indices]

                # Sort the values and select the smallest p-fraction
                sorted_values, sorted_indices = torch.sort(edge_values)
                num_to_select = max(1, int(p * len(sorted_values)))  # At least one edge to select

                # Select the p-fraction smallest edges
                selected_indices = indices[sorted_indices[:num_to_select]]

                # Store the selected (i, j) indices for later update
                for j in selected_indices:
                    updates_to_set.append((i, j))

        # Update A_prev to be the current A^t for the next iteration
        A_prev = A_current.clone()

    #Convert updates_to_set into edge_index format
    updates_to_set = torch.tensor(updates_to_set, device=edge_index.device).T  # Convert to (2, E) shape

    #Combine original edge_index with new edges
    updated_edge_index = torch.cat([edge_index, updates_to_set], dim=1)

    # Remove duplicate edges to ensure unique edge_index
    updated_edge_index = torch.unique(updated_edge_index, dim=1)

    return updated_edge_index





