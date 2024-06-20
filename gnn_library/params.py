from .OBM_GENConv import OBM_GENConv
from .OBM_deeperGCN import DeeperGCN
from .OBM_GATv2Conv import OBM_GATv2Conv
from .OBM_GraphConv import OBM_GraphConv
from .OBM_GCNConv import OBM_GCNConv

NETWORKS = {
    'GENConv': OBM_GENConv,
    'DeeperGCN': DeeperGCN,
    'GATv2Conv': OBM_GATv2Conv,
    'GraphConv': OBM_GraphConv,
    'GCNConv': OBM_GCNConv
}

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
        'hidden_dim',
        'output_dim',
        'dropout',
        'device'
    ]
}

GRAPH_CONFIGS = [

]