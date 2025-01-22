import torch
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


import torch


def find_elements_larger_than(matrix, a):
    """
    Find the indices of elements larger than a given threshold in an n x n matrix.

    Args:
        matrix (torch.Tensor): An n x n PyTorch matrix.
        a (float): The threshold value.

    Returns:
        indices (torch.Tensor): A 2 x m tensor where each column contains the row and column index
                                of one of the elements larger than 'a'.
    """
    # Step 1: Apply the mask to find elements larger than 'a'
    mask = matrix > a

    # Step 2: Get the indices of elements satisfying the condition
    indices = torch.nonzero(mask, as_tuple=False).T

    return indices


import torch


def top_m_entries_indices(matrix, m):
    """
    Find the indices of the top m entries in an n*n matrix.

    Args:
        matrix (torch.Tensor): An n x n PyTorch matrix.
        m (int): The number of maximum entries to retrieve.

    Returns:
        indices (torch.Tensor): A 2 x m tensor where each column contains the row and column index
                                of one of the top m entries.
    """
    # Step 1: Flatten the matrix into a 1D array
    flat_matrix = matrix.flatten()

    # Step 2: Find the top m values' indices in the flattened matrix
    top_values, top_indices = torch.topk(flat_matrix, m)

    # Step 3: Convert flat indices back to 2D indices
    rows = top_indices // matrix.size(1)  # Integer division for row indices
    cols = top_indices % matrix.size(1)  # Modulo operation for column indices

    # Step 4: Stack the row and column indices into a 2 x m matrix
    indices = torch.stack([rows, cols])

    return indices


# Example usage:
n = 5  # Size of the matrix
m = 3  # Number of maximum entries to find

matrix = torch.rand(n, n)  # Create a random n x n matrix

indices = top_m_entries_indices(matrix, m)
print("Matrix:\n", matrix)
print("Top m entries' indices (2 x m):\n", indices)

# Example usage:
n = 5  # Size of the matrix
a = 0.7  # Threshold value

matrix = torch.rand(n, n)  # Create a random n x n matrix

indices = find_elements_larger_than(matrix, a)
print("Matrix:\n", matrix)
print(f"Elements larger than {a} are at indices (2 x m):\n", indices)

