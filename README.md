# GOKU: Mitigating Over-Squashing in GNNs by Spectrum-Preserving Sparsification

Official implementation of the ICML 2025 paper: "Mitigating Over-Squashing in Graph Neural Networks by Spectrum-Preserving Sparsification"


## Installation

```bash
# Clone the repository
git clone https://github.com/your-username/goku.git
cd goku

conda create -n goku python=3.8
conda activate goku

# Install dependencies
pip install -r requirements.txt

# Alternatively, if you have a environment.yml file
# conda env update -f environment.yml
```

## Usage

The repository provides two main scripts:

### Graph Classification

```bash
python run_graph_classification.py --dataset [DATASET] --rewiring goku --k_guess [K_GUESS] --beta [BETA]
```

### Node Classification

```bash
python run_node_classification.py --dataset [DATASET] --rewiring goku --k_guess [K_GUESS] --beta [BETA]
```


## Citation

If you find this code useful, please cite our paper:

```bibtex
@inproceedings{liang2025mitigating,
  title={Mitigating Over-Squashing in Graph Neural Networks by Spectrum-Preserving Sparsification},
  author={Liang, Langzhang and Bu, Fanchen and Song, Zixing and Xu, Zenglin and Pan, Shirui and Shin, Kijung},
  booktitle={Proceedings of the 42nd International Conference on Machine Learning (ICML)},
  year={2025}
}
```

## Acknowledgements

The baseline code in this repository is adapted from "Revisiting Over-smoothing and Over-squashing Using Ollivier-Ricci Curvature" (ICML2023)
