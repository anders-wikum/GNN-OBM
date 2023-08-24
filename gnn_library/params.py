from .OBM_GENConv import OBM_GENConv
from .OBM_NNConv import OBM_NNConv
from .OBM_multiclass import GraphClassifier


NETWORKS = {
    'GENConv': OBM_GENConv,
    'NNConv': OBM_NNConv,
    'NN': GraphClassifier
}

#TODO: Make this an absolute path so that it can be called from anywhere
MODEL_SAVE_FOLDER = './trained_models/'
REQ_ARGS = {
    'regression': [
        'processor',
        'head',
        'num_layers',
        'aggr',
        'node_feature_dim',
        'edge_feature_dim',
        'graph_feature_dim',
        'hidden_dim',
        'output_dim',
        'dropout',
        'device'
    ],
    'classification': [
        'processor',
        'head',
        'num_layers',
        'aggr',
        'node_feature_dim',
        'edge_feature_dim',
        'graph_feature_dim',
        'hidden_dim',
        'output_dim',
        'dropout',
        'device'
    ],
    'meta': [
        'processor',
        'head',
        'num_layers',
        'input_dim',
        'hidden_dim',
        'output_dim',
        'dropout',
        'device'
    ]
}