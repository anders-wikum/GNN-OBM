import numpy as np

_Array = np.ndarray

SAMPLER_SPECS = {
    'GEOM': set(['threshold', 'scaling']),
    'ER': set(['p', 'weighted']),
    'BA': set(['ba_param', 'weighted']),
    'COMP': set([])
}

GRAPH_TYPES = ['GEOM', 'ER', 'BA', 'COMP']
