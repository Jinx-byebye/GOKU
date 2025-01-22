from attrdict import AttrDict
from torch_geometric.datasets import WebKB, WikipediaNetwork, Actor, Planetoid
from torch_geometric.utils import to_networkx, from_networkx, to_undirected
from torch_geometric.transforms import LargestConnectedComponents, ToUndirected
from experiments.node_classification import Experiment

import time
import torch
import numpy as np
import pandas as pd
from hyperparams import get_args_from_input
from preprocessing import rewiring, sdrf, fosr, borf, goku_rewiring, delaunay_rewiring, laser_rewiring
from preprocessing.gtr import PrecomputeGTREdges, AddPrecomputedGTREdges, AddGTREdges

import nni
from nni.utils import merge_parameter

largest_cc = LargestConnectedComponents()
cornell = WebKB(root="data", name="Cornell")
wisconsin = WebKB(root="data", name="Wisconsin")
texas = WebKB(root="data", name="Texas")
chameleon = WikipediaNetwork(root="data", name="chameleon")
cora = Planetoid(root="data", name="cora")
citeseer = Planetoid(root="data", name="citeseer")
datasets = {"cornell": cornell, "wisconsin": wisconsin, "texas": texas, 
        "chameleon": chameleon,
        "cora": cora, "citeseer": citeseer}

for key in datasets:
    dataset = datasets[key]
    dataset.data.edge_index = to_undirected(dataset.data.edge_index)

def log_to_file(message, filename="results/node_classification.txt"):
    print(message)
    file = open(filename, "a")
    file.write(message)
    file.close()

default_args = AttrDict({
    "dropout": 0.,
    "num_layers": 3,
    "hidden_dim": 128,
    "learning_rate": 1e-3,
    "layer_type": "GCN",
    "display": True,
    "num_trials": 10,
    "eval_every": 1,
    "rewiring": "goku",
    "num_iterations": 50,
    "patience": 100,
    "dataset": "cora",
    "borf_batch_add" : 20,
    "borf_batch_remove" : 10,
    "sdrf_remove_edges" : False,
    "num_relations": 1,
    "epsilon": 0.1,
    "er_est_method": "kts",
    "to_undirected": True,
    "metric": "degree",
    "k_guess": 20,
    "step_size": 0.001,
    "beta": 1,
})


results = []
args = default_args
args += get_args_from_input()

optimized_params = nni.get_next_parameter()
args = merge_parameter(args, optimized_params)

if args.rewiring == 'gtr':
    num_edges_to_add = 10
    rewiring_transform = AddGTREdges(num_edges=num_edges_to_add, try_gpu=True)

if args.dataset:
    # restricts to just the given dataset if this mode is chosen
    name = args.dataset
    datasets = {name: datasets[name]}

for key in datasets:
    accuracies = []
    print(f"TESTING: {key} ({args.rewiring})")
    dataset = datasets[key]

    start = time.time()
    if args.rewiring == "fosr":
        edge_index, edge_type, _ = fosr.edge_rewire(dataset.data.edge_index.numpy(), num_iterations=args.num_iterations)
        dataset.data.edge_index = torch.tensor(edge_index)
        dataset.data.edge_type = torch.tensor(edge_type)
        print(dataset.data.num_edges)
        print(len(dataset.data.edge_type))
    elif args.rewiring == "sdrf_bfc":
        curvature_type = "bfc"
        dataset.data.edge_index, dataset.data.edge_type = sdrf.sdrf(dataset.data, loops=args.num_iterations, remove_edges=args.sdrf_remove_edges, 
                is_undirected=True, curvature=curvature_type)
    elif args.rewiring == "borf":
        print(f"[INFO] BORF hyper-parameter : num_iterations = {args.num_iterations}")
        print(f"[INFO] BORF hyper-parameter : batch_add = {args.borf_batch_add}")
        print(f"[INFO] BORF hyper-parameter : num_iterations = {args.borf_batch_remove}")
        dataset.data.edge_index, dataset.data.edge_type = borf.borf3(dataset.data, 
                loops=args.num_iterations, 
                remove_edges=False, 
                is_undirected=True,
                batch_add=args.borf_batch_add,
                batch_remove=args.borf_batch_remove,
                dataset_name=key,
                graph_index=0)
        print(len(dataset.data.edge_type))
    elif args.rewiring == "sdrf_orc":
        curvature_type = "orc"
        dataset.data.edge_index, dataset.data.edge_type = sdrf.sdrf(dataset.data, loops=args.num_iterations, remove_edges=False, 
                is_undirected=True, curvature=curvature_type)
    elif args.rewiring == 'goku':
        dataset.data.edge_index, dataset.data.edge_type, dataset.data.edge_weight = goku_rewiring.goku(
            dataset.data.edge_index.numpy().transpose(), dataset.data.x,
            to_undirected=args.to_undirected,
            k_guess=args.k_guess, step_size=args.step_size,
            num_relations=args.num_relations,
            device="cuda:0" if torch.cuda.is_available() else "cpu",
            metric=args.metric,
            beta=args.beta
            )

    elif args.rewiring == 'delaunay':
        dataset.data.edge_index = delaunay_rewiring.dalaunay(dataset.data.x).long()
        dataset.data.edge_type = torch.zeros_like(dataset.data.edge_index[0], dtype=torch.int64)
    elif args.rewiring == 'gtr':
        dataset.data = rewiring_transform(dataset.data)
    elif args.rewiring == 'laser':
        dataset.data.edge_index = laser_rewiring.laser(dataset.data.edge_index, p=0.15, max_k=3).long()


    end = time.time()
    rewiring_duration = end - start

    # print(rewiring.spectral_gap(to_networkx(dataset.data, to_undirected=True)))
    start = time.time()
    for trial in range(args.num_trials):
        print(f"TRIAL #{trial+1}")
        test_accs = []
        for i in range(args.num_splits):
            train_acc, validation_acc, test_acc = Experiment(args=args, dataset=dataset).run()
            test_accs.append(test_acc)
        test_acc = max(test_accs)
        nni.report_intermediate_result(test_acc)
        accuracies.append(test_acc)
    end = time.time()
    run_duration = end - start

    log_to_file(f"RESULTS FOR {key} ({args.rewiring}):\n")
    log_to_file(f"average acc: {np.mean(accuracies)}\n")
    nni.report_final_result(np.mean(accuracies))
    log_to_file(f"plus/minus:  {2 * np.std(accuracies)/(args.num_trials ** 0.5)}\n\n")
    results.append({
        "dataset": key,
        "rewiring": args.rewiring,
        "num_iterations": args.num_iterations,
        "borf_batch_add" : args.borf_batch_add,
        "borf_batch_remove" : args.borf_batch_remove,
        "avg_accuracy": np.mean(accuracies),
        "ci":  2 * np.std(accuracies)/(args.num_trials ** 0.5),
        "run_duration" : run_duration,
        "rewiring_duration" : rewiring_duration
    })
    results_df = pd.DataFrame(results)
    with open(f'results/node_classification_{args.layer_type}_{args.rewiring}.csv', 'a') as f:
        results_df.to_csv(f, mode='a', header=f.tell()==0)
