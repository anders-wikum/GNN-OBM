import numpy as np

_Array = np.ndarray

GRAPH_TYPES = ['GEOM', 'ER', 'BA', 'COMP', 'GM']

SAMPLER_SPECS = {
    'GEOM': set(['threshold', 'scaling']),
    'ER': set(['p', 'weighted']),
    'BA': set(['ba_param', 'weighted']),
    'COMP': set([]),
    'GM': set([])
}
