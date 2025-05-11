from attrdict import AttrDict
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import to_networkx, from_networkx, to_dense_adj
from experiments.graph_classification import Experiment

import time
import tqdm
import torch
import numpy as np
import pandas as pd
from hyperparams import get_args_from_input
from preprocessing import rewiring, sdrf, fosr, digl, borf, goku_rewiring, delaunay_rewiring, laser_rewiring
from preprocessing.gtr import PrecomputeGTREdges, AddPrecomputedGTREdges, AddGTREdges


def calculate_average_spectral_gap(dataset):
    """
    Calculate the average spectral gap across all graphs in a dataset.
    
    Args:
        dataset (list): List of graph data objects
        
    Returns:
        float: Average spectral gap value
    """
    spectral_gaps = []
    for graph in dataset:
        G = to_networkx(graph, to_undirected=True)
        spectral_gap = rewiring.spectral_gap(G)
        spectral_gaps.append(spectral_gap)
    return sum(spectral_gaps) / len(spectral_gaps)


def log_to_file(message, filename="results/graph_classification.txt"):
    """
    Log a message to both console and a file.
    
    Args:
        message (str): The message to log
        filename (str): Path to the log file
    """
    print(message)
    with open(filename, "a") as file:
        file.write(message)


# Default arguments for the experiment
default_args = AttrDict({
    "dropout": 0.,
    "num_layers": 4,
    "hidden_dim": 64,
    "learning_rate": 1e-3,
    "layer_type": "GCN",
    "display": True,
    "num_trials": 100,
    "eval_every": 1,
    "rewiring": "goku",
    "num_iterations": 10,
    "patience": 100,
    "output_dim": 2,
    "alpha": 0.1,
    "eps": 0.001,
    "dataset": "mutag",
    "alpha_dim": 10,
    "eps_dim": 10,
    "num_heads": 8,
    "num_out_heads": 1,
    "num_layers_output": 1,
    "residual": False,
    "in_feat_dropout": 0.0,
    "dropout_attn": 0.0,
    "activation": "relu",
    "last_layer_fa": False,
    "borf_batch_add": 4,
    "borf_batch_remove": 2,
    "sdrf_remove_edges": False,
    "num_relations": 1,
    "epsilon": 0.1, # er approximation error
    "er_est_method": "kts",
    "to_undirected": True,
    "metric": "degree",
    "mini_k": 20, # minimum edges to add
    "step_size": 0.01,
    "beta": 1.0 # #edges ratio between output and original graphs
})

# Dataset-specific hyperparameters
dataset_hyperparams = {
    "mutag": AttrDict({"output_dim": 2}),
    "enzymes": AttrDict({"output_dim": 6}),
    "proteins": AttrDict({"output_dim": 2}),
    "collab": AttrDict({"output_dim": 3}),
    "imdb": AttrDict({"output_dim": 2}),
    "reddit": AttrDict({"output_dim": 2})
}


# Main execution
results = []
args = default_args
args += get_args_from_input()  # Override defaults with command line inputs

# Load datasets
mutag = list(TUDataset(root="data", name="MUTAG"))
enzymes = list(TUDataset(root="data", name="ENZYMES"))
proteins = list(TUDataset(root="data", name="PROTEINS"))
imdb = list(TUDataset(root="data", name="IMDB-BINARY"))
datasets = {"mutag": mutag, "enzymes": enzymes, "imdb": imdb, "proteins": proteins}

# Add node features for datasets that don't have them
for key in datasets:
    if key in ["reddit", "imdb", "collab"]:
        for graph in datasets[key]:
            num_nodes = graph.num_nodes
            graph.x = torch.ones((num_nodes, 1))  # Add simple node features (all ones)

# Initialize GTR rewiring if selected
if args.rewiring == 'gtr':
    num_edges_to_add = 10
    rewiring_transform = AddGTREdges(num_edges=num_edges_to_add, try_gpu=True)

# Restrict to a single dataset if specified
if args.dataset:
    dataset_name = args.dataset
    datasets = {dataset_name: datasets[dataset_name]}

# Process each dataset
for dataset_name in datasets:
    # Update arguments with dataset-specific hyperparameters
    args += dataset_hyperparams[dataset_name]
    
    train_accuracies = []
    validation_accuracies = []
    test_accuracies = []
    energies = []
    
    print(f"TESTING: {dataset_name} ({args.rewiring} - layer {args.layer_type})")
    dataset = datasets[dataset_name]

    # Perform rewiring on the dataset
    print('REWIRING STARTED...')
    start_time = time.time()
    
    with tqdm.tqdm(total=len(dataset)) as progress_bar:
        if args.rewiring == "fosr":
            # Factorizable Operator Spectral Rewiring
            for i in range(len(dataset)):
                edge_index, edge_type, _ = fosr.edge_rewire(
                    dataset[i].edge_index.numpy(), 
                    num_iterations=args.num_iterations
                )
                dataset[i].edge_index = torch.tensor(edge_index)
                dataset[i].edge_type = torch.tensor(edge_type)
                progress_bar.update(1)
                
        elif args.rewiring == "sdrf_orc":
            # Stochastic Discrete Ricci Flow with Ollivier-Ricci Curvature
            for i in range(len(dataset)):
                dataset[i].edge_index, dataset[i].edge_type = sdrf.sdrf(
                    dataset[i], 
                    loops=args.num_iterations, 
                    remove_edges=False, 
                    is_undirected=True, 
                    curvature='orc'
                )
                progress_bar.update(1)
                
        elif args.rewiring == "sdrf_bfc":
            # Stochastic Discrete Ricci Flow with Balanced Forman Curvature
            for i in range(len(dataset)):
                dataset[i].edge_index, dataset[i].edge_type = sdrf.sdrf(
                    dataset[i], 
                    loops=args.num_iterations, 
                    remove_edges=args["sdrf_remove_edges"], 
                    is_undirected=True, 
                    curvature='bfc'
                )
                progress_bar.update(1)
                
        elif args.rewiring == "borf":
            # Balanced Optimal Transport Rewiring Framework
            print(f"[INFO] BORF hyper-parameter : num_iterations = {args.num_iterations}")
            print(f"[INFO] BORF hyper-parameter : batch_add = {args.borf_batch_add}")
            print(f"[INFO] BORF hyper-parameter : batch_remove = {args.borf_batch_remove}")
            
            for i in range(len(dataset)):
                dataset[i].edge_index, dataset[i].edge_type = borf.borf3(
                    dataset[i], 
                    loops=args.num_iterations, 
                    remove_edges=False, 
                    is_undirected=True,
                    batch_add=args.borf_batch_add,
                    batch_remove=args.borf_batch_remove,
                    dataset_name=dataset_name,
                    graph_index=i
                )
                progress_bar.update(1)

        elif args.rewiring == "digl":
            # Distance Encoding Improved Graph Learning
            for i in range(len(dataset)):
                dataset[i].edge_index = digl.rewire(
                    dataset[i], 
                    alpha=0.1, 
                    eps=0.05
                )
                num_edges = dataset[i].edge_index.shape[1]
                dataset[i].edge_type = torch.tensor(np.zeros(num_edges, dtype=np.int64))
                progress_bar.update(1)
                
        elif args.rewiring == "goku":
            # Graph Optimization through Kombinatorial Unification
            print(f'GOKU hyperparameter : epsilon (estimation error for ER) = {args.epsilon}')
            print(f'GOKU hyperparameter : k = {args.mini_k}')
            print(f'GOKU hyperparameter : num relations for mapping weighted edges to unweighted edges = {args.num_relations}')
            
            for i in range(len(dataset)):
                print(f'Rewiring {i+1}-th graph.')
                dataset[i].edge_index, dataset[i].edge_type, dataset[i].edge_weight = goku_rewiring.goku(
                    dataset[i].edge_index.numpy().transpose(), 
                    dataset[i].x,
                    to_undirected=args.to_undirected,
                    mini_k=args.mini_k,
                    step_size=args.step_size,
                    num_relations=args.num_relations,
                    device="cuda:0" if torch.cuda.is_available() else "cpu",
                    beta=args.beta
                )
                progress_bar.update(1)
                
        elif args.rewiring == 'delaunay':
            # Delaunay triangulation based rewiring
            for i in range(len(dataset)):
                dataset[i].edge_index = delaunay_rewiring.dalaunay(dataset[i].x).long()
                dataset[i].edge_type = torch.zeros_like(dataset[i].edge_index[0], dtype=torch.int64)
                progress_bar.update(1)

        elif args.rewiring == 'gtr':
            # Graph Transformer Rewiring
            for i in range(len(dataset)):
                dataset[i] = rewiring_transform(dataset[i])
                progress_bar.update(1)
                
        elif args.rewiring == 'laser':
            # LASER rewiring method
            for i in range(len(dataset)):
                dataset[i].edge_index = laser_rewiring.laser(
                    dataset[i].edge_index, 
                    p=0.15, 
                    max_k=3
                ).long()
                dataset[i].edge_type = torch.zeros_like(dataset[i].edge_index[0], dtype=torch.int64)
                progress_bar.update(1)
                
    end_time = time.time()
    rewiring_duration = end_time - start_time
    print(f'Duration of rewiring: {rewiring_duration} seconds')

    # Train and evaluate models
    print('TRAINING STARTED...')
    start_time = time.time()
    
    for trial in range(args.num_trials):
        train_acc, validation_acc, test_acc, energy = Experiment(args=args, dataset=dataset).run()
        train_accuracies.append(train_acc)
        validation_accuracies.append(validation_acc)
        test_accuracies.append(test_acc)
        energies.append(energy)
        
    end_time = time.time()
    run_duration = end_time - start_time

    # Calculate statistics
    train_mean = 100 * np.mean(train_accuracies)
    val_mean = 100 * np.mean(validation_accuracies)
    test_mean = 100 * np.mean(test_accuracies)
    energy_mean = 100 * np.mean(energies)
    
    # Calculate confidence intervals (95%)
    train_ci = 2 * np.std(train_accuracies)/(args.num_trials ** 0.5)
    val_ci = 2 * np.std(validation_accuracies)/(args.num_trials ** 0.5)
    test_ci = 2 * np.std(test_accuracies)/(args.num_trials ** 0.5)
    energy_ci = 200 * np.std(energies)/(args.num_trials ** 0.5)
    
    # Log results
    # log_to_file(f"RESULTS FOR {dataset_name} ({args.rewiring}):\n")
    # log_to_file(f"average acc: {test_mean}\n")
    # log_to_file(f"plus/minus:  {test_ci}\n\n")
    
    # Store results for CSV output
    results.append({
        "dataset": dataset_name,
        "rewiring": args.rewiring,
        "layer_type": args.layer_type,
        "num_iterations": args.num_iterations,
        "borf_batch_add": args.borf_batch_add,
        "borf_batch_remove": args.borf_batch_remove,
        "sdrf_remove_edges": args.sdrf_remove_edges, 
        "alpha": args.alpha,
        "eps": args.eps,
        "test_mean": test_mean,
        "test_ci": test_ci,
        "val_mean": val_mean,
        "val_ci": val_ci,
        "train_mean": train_mean,
        "train_ci": train_ci,
        "energy_mean": energy_mean,
        "energy_ci": energy_ci,
        "last_layer_fa": args.last_layer_fa,
        "rewiring_duration": rewiring_duration,
        "run_duration": run_duration,
    })

    results_df = pd.DataFrame(results)
    csv_path = f'results/graph_classification_{args.layer_type}_{args.rewiring}.csv'
    print(f'Test mean acc: {results_df["test_mean"]}')
    # with open(csv_path, 'a') as f:
    #     results_df.to_csv(f, mode='a', header=f.tell()==0)
