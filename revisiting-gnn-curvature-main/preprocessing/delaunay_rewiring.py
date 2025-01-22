import umap
import torch
import networkx as nx
from scipy.spatial import Delaunay
from torch_geometric.utils import to_undirected

def dalaunay(x):
    x = x.cpu().numpy()
    position = umap.UMAP(n_components=2).fit_transform(x)
    delaunay = Delaunay(position, qhull_options='QJ')

    edges = []
    for simplex in delaunay.simplices:
        for i in range(3):
            for j in range(i + 1, 3):
                edge = (simplex[i], simplex[j])
                edgess = (simplex[j], simplex[i])
                edges.append(edge)
                edges.append(edgess)

    delaunay_graph = nx.Graph(edges)

    edge_index = torch.tensor(list(delaunay_graph.edges)).t().contiguous()
    return to_undirected(edge_index)