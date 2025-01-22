import torch
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
from torch_geometric.utils import to_scipy_sparse_matrix


def compute_laplacian_and_eigenvalues(edge_index, edge_weight=None, num_nodes=None, use_gpu=True):
    # Ensure data is on GPU if required
    device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
    edge_index = edge_index.to(device)
    if edge_weight is None:
        edge_weight = torch.ones(edge_index.shape[1], dtype=torch.float, device=device)
    else:
        edge_weight = edge_weight.to(device)

    # Create the adjacency matrix A (sparse representation)
    adj = torch.sparse_coo_tensor(edge_index, edge_weight, (num_nodes, num_nodes))

    # Convert to scipy sparse matrix for easier manipulation (must be on CPU)
    adj_scipy = to_scipy_sparse_matrix(adj.cpu())

    # Compute degree matrix D
    degree = np.array(adj_scipy.sum(axis=1)).flatten()
    D = sp.diags(degree, offsets=0)  # Diagonal matrix

    # Compute the Laplacian matrix L = D - A
    L = D - adj_scipy

    # Compute the eigenvalues of the Laplacian
    eigenvalues = np.linalg.eigvals(L.toarray())

    # Visualize the eigenvalues with a color bar
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(range(len(eigenvalues)), np.sort(eigenvalues), c=np.sort(eigenvalues), cmap='viridis', s=20)
    plt.colorbar(scatter, label='Eigenvalue Magnitude')
    plt.title('Eigenvalue Distribution of Laplacian Matrix')
    plt.xlabel('Index (Sorted)')
    plt.ylabel('Eigenvalue')
    plt.grid(True)
    plt.show()

    return eigenvalues


# Example usage
edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]])  # Example edge index (2 x E)
edge_weight = torch.tensor([1.0, 1.0, 1.0, 1.0])  # Example edge weights (1 for each edge)
num_nodes = 4  # Number of nodes

eigenvalues = compute_laplacian_and_eigenvalues(edge_index, edge_weight, num_nodes, use_gpu=True)
print("Eigenvalues:", eigenvalues)
