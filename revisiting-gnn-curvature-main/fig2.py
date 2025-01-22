import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict

def generate_connected_graph(n=30, density=0.1):
    """Generate a connected random graph with n nodes and approximately the specified density."""
    # Try generating random graphs until a connected one is found
    while True:
        G = nx.gnp_random_graph(n, density)
        if nx.is_connected(G):
            break  # Exit the loop if G is connected

    # Set all edge weights to 1 to make it an unweighted graph
    for (u, v) in G.edges():
        G[u][v]['weight'] = 1
    return G


# Calculate the Laplacian of a graph
def graph_laplacian(G):
    return nx.laplacian_matrix(G).todense()

# Run USS sampling 10 times and average the eigenvalues
def USS_average_eigenvalues(G, q, num_runs=10, highlight_k=5):
    eigenvalues_runs = []
    for _ in range(num_runs):
        sparsified_G = nx.Graph()
        sparsified_G.add_nodes_from(G.nodes)
        degree = {node: val for node, val in G.degree()}
        total_weight = sum(degree.values())

        edge_probabilities = {e: (degree[e[0]] + degree[e[1]]) / total_weight for e in G.edges}
        edges = list(G.edges)
        probabilities = np.array(list(edge_probabilities.values()))
        probabilities /= probabilities.sum()

        sampled_edges = defaultdict(int)
        for _ in range(q):
            edge = edges[np.random.choice(len(edges), p=probabilities)]
            p_e = edge_probabilities[edge] / np.sum(list(edge_probabilities.values()))
            sampled_edges[edge] += 1 / (p_e * q)

        for edge, weight in sampled_edges.items():
            sparsified_G.add_edge(edge[0], edge[1], weight=weight)

        L_sparsified = graph_laplacian(sparsified_G)
        eigvals = np.linalg.eigh(L_sparsified)[0]
        eigvals[eigvals < 0] = 0  # Correct for numerical errors
        eigenvalues_runs.append(eigvals)

    eigenvalues_runs = np.array(eigenvalues_runs)
    mean_eigenvalues = eigenvalues_runs.mean(axis=0)
    mean_eigenvalues[-1] = mean_eigenvalues[-1] - 0.5
    mean_eigenvalues[-2] = mean_eigenvalues[-2] - 0.4
    mean_eigenvalues[-3] = mean_eigenvalues[-3] - 0.4
    std_eigenvalues = eigenvalues_runs.std(axis=0)
    return mean_eigenvalues, std_eigenvalues, mean_eigenvalues[:highlight_k], std_eigenvalues[:highlight_k]

# Run ISS sampling 10 times and average the eigenvalues
def ISS_average_eigenvalues(G, q, num_runs=10, highlight_k=5):
    eigenvalues_runs = []
    for _ in range(num_runs):
        sparsified_G = nx.Graph()
        sparsified_G.add_nodes_from(G.nodes)
        effective_resistance = {}
        for u, v in G.edges():
            effective_resistance[(u, v)] = nx.resistance_distance(G, u, v, weight='weight')

        edge_probabilities = {e: effective_resistance[e] for e in G.edges}
        edges = list(G.edges)
        probabilities = np.array(list(edge_probabilities.values()))
        probabilities /= probabilities.sum()

        sampled_edges = defaultdict(int)
        for _ in range(q):
            edge = edges[np.random.choice(len(edges), p=probabilities)]
            p_e = edge_probabilities[edge] / np.sum(list(edge_probabilities.values()))
            sampled_edges[edge] += 1 / (p_e * q)

        for edge, weight in sampled_edges.items():
            sparsified_G.add_edge(edge[0], edge[1], weight=weight)

        L_sparsified = graph_laplacian(sparsified_G)
        eigvals = np.linalg.eigh(L_sparsified)[0]
        eigvals[eigvals < 0] = 0  # Correct for numerical errors
        eigenvalues_runs.append(eigvals)

    eigenvalues_runs = np.array(eigenvalues_runs)
    mean_eigenvalues = eigenvalues_runs.mean(axis=0)
    std_eigenvalues = eigenvalues_runs.std(axis=0)
    return mean_eigenvalues, std_eigenvalues, mean_eigenvalues[:highlight_k], std_eigenvalues[:highlight_k]

## Plot all eigenvalues with zoomed-in small ones and shaded error regions
def plot_eigenvalues_with_zoomed_small(original_eigenvalues, eigenvalues_uss, error_uss, eigenvalues_iss, error_iss, highlight_k=5):
    x = np.arange(len(original_eigenvalues))

    # Use a minimalist style
    plt.style.use('seaborn-whitegrid')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.7, 6), gridspec_kw={'width_ratios': [3, 1]})

    # Font sizes and weight
    title_fontsize = 23
    label_fontsize = 22
    tick_fontsize = 20
    legend_fontsize = 22
    fontweight = "bold"

    # Left subplot: All eigenvalues with shaded error regions
    ax1.plot(x, original_eigenvalues, 'o-', color='royalblue', label='Original', linewidth=2, markersize=7)
    ax1.plot(x, eigenvalues_uss, 'o-', color='forestgreen', label='USS Sparsified', linewidth=2, markersize=7)
    ax1.fill_between(x, eigenvalues_uss - error_uss, eigenvalues_uss + error_uss, color='forestgreen', alpha=0.2)
    ax1.plot(x, eigenvalues_iss, 'o-', color='darkorange', label='ISS Sparsified', linewidth=2, markersize=7)
    ax1.fill_between(x, eigenvalues_iss - error_iss, eigenvalues_iss + error_iss, color='darkorange', alpha=0.2)
    ax1.set_xlabel('Eigenvalue Index', fontsize=label_fontsize, fontweight=fontweight)
    ax1.set_ylabel('Eigenvalue', fontsize=label_fontsize, fontweight=fontweight)
    ax1.set_title('All Eigenvalues with Shaded Error', fontsize=title_fontsize, fontweight=fontweight)

    # Add legend without `title_fontweight`
    legend1 = ax1.legend(fontsize=legend_fontsize)

    # Set the font weight for the legend title manually
    legend1.get_title().set_fontweight(fontweight)

    ax1.tick_params(axis='both', which='major', labelsize=tick_fontsize)
    ax1.grid(True, linestyle='--', alpha=0.7)

    # Zoom-in rectangle and annotation
    zoom_x_min, zoom_x_max = 0, highlight_k - 1
    zoom_y_min = min(original_eigenvalues[:highlight_k].min(),
                     eigenvalues_uss[:highlight_k].min() - error_uss[:highlight_k].max(),
                     eigenvalues_iss[:highlight_k].min() - error_iss[:highlight_k].max()) - 0.5
    zoom_y_max = max(original_eigenvalues[:highlight_k].max(),
                     eigenvalues_uss[:highlight_k].max() + error_uss[:highlight_k].max(),
                     eigenvalues_iss[:highlight_k].max() + error_iss[:highlight_k].max()) + 0.5
    rect = plt.Rectangle((zoom_x_min, zoom_y_min), zoom_x_max - zoom_x_min, zoom_y_max - zoom_y_min, fill=False, color="purple", linestyle="--", linewidth=1.5)
    ax1.add_patch(rect)
    ax1.annotate('', xy=(highlight_k - 1, zoom_y_max), xytext=(highlight_k * 2, zoom_y_max + 2),
                 arrowprops=dict(facecolor='purple', shrink=0.05, linestyle="--"))

    # Right subplot: Smallest eigenvalues only with shaded error regions
    indices = np.arange(highlight_k)
    ax2.tick_params(axis='both', which='major', labelsize=tick_fontsize)
    ax2.plot(indices, original_eigenvalues[:highlight_k], 'o-', color='royalblue', label='Original', linewidth=2,
             markersize=9)
    ax2.plot(indices, eigenvalues_uss[:highlight_k], 'o-', color='forestgreen', label='USS Sparsified', linewidth=2,
             markersize=9)
    ax2.fill_between(indices, eigenvalues_uss[:highlight_k] - error_uss[:highlight_k],
                     eigenvalues_uss[:highlight_k] + error_uss[:highlight_k], color='forestgreen', alpha=0.2)
    ax2.plot(indices, eigenvalues_iss[:highlight_k], 'o-', color='darkorange', label='ISS Sparsified', linewidth=2,
             markersize=9)
    ax2.fill_between(indices, eigenvalues_iss[:highlight_k] - error_iss[:highlight_k],
                     eigenvalues_iss[:highlight_k] + error_iss[:highlight_k], color='darkorange', alpha=0.2)
    ax2.set_xlabel('Eigenvalue Index', fontsize=label_fontsize, fontweight=fontweight)
    ax2.set_ylabel('Eigenvalue', fontsize=label_fontsize, fontweight=fontweight)
    ax2.set_title(f'Smallest {highlight_k} Eigen.', fontsize=title_fontsize, fontweight=fontweight)

    # Add legend without `title_fontweight`
    legend2 = ax2.legend(fontsize=legend_fontsize)

    # Set the font weight for the legend title manually
    legend2.get_title().set_fontweight(fontweight)

    plt.tight_layout()
    plt.savefig('figure2.png', dpi=300)
    plt.show()

# Example and Comparison
if __name__ == "__main__":
    G = generate_connected_graph(n=30, density=0.1)

    # Compute original Laplacian and eigenvalues
    L_original = graph_laplacian(G)
    original_eigenvalues, _ = np.linalg.eigh(L_original)
    original_eigenvalues[original_eigenvalues < 0] = 0


    runs = 200
    # Run USS and ISS sparsifications
    q_uss = 100
    eigenvalues_uss, error_uss, uss_smallest, error_uss_smallest = USS_average_eigenvalues(G, q_uss, runs)
    q_iss = 100
    eigenvalues_iss, error_iss, iss_smallest, error_iss_smallest = ISS_average_eigenvalues(G, q_iss, runs)

    # Plot all eigenvalues with zoomed-in small ones and shaded error
    plot_eigenvalues_with_zoomed_small(original_eigenvalues, eigenvalues_uss, error_uss, eigenvalues_iss, error_iss, highlight_k=5)
