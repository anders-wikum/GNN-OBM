# MAGNOLIA: Matching Algorithms via GNNs for Online Value-to-go Approximation

Please run the cells in `notebooks/osmnx_generator.ipynb`to enable generation 
of graph inputs from OSMnx -- this may take a few minutes.

## Repository overview

Code for all experiments can be found in the `/notebooks/` directory. Experiment 
files follow the naming convention "exp_[experiment name]". GNN and greedy-t 
hyperparameter tuning can be found in `/notebooks/gnn_tuner.ipynb` and 
`/notebooks/greedy_tuner.ipynb`, respectively.

`/gnn_library/` contains all information relevant to defining, training, testing,
and saving GNN models. Several pre-trained models can be loaded from
`/trained_models/'. The experiments contain many examples of loading
these models.

Finally, `instance_generator.py` is used for the generation of OBBM instances
of different graph configurations, `torch_converter.py` gives code for
converting instances into PyG representations, `evaluate.py` contains code
for estimating average competitive ratio for learned models, and `algorithms.py`
contains implementations of baseline algorithms.
