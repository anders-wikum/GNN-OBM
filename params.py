import numpy as np
from typing import Tuple

_Array = np.ndarray

# An instance is: A, p, noisy A, noisy p
_Instance = Tuple[_Array, _Array, _Array, _Array]

# ============== Graph sampler params ===================== #

GRAPH_TYPES = ['GEOM', 'ER', 'BA', 'COMP', 'GM', 'FEAT', 'PART', 'OSMNX']

SAMPLER_SPECS = {
    'GEOM': set(['threshold', 'scaling']),
    'ER': set(['p', 'weighted']),
    'BA': set(['ba_param', 'weighted']),
    'COMP': set([]),
    'GM': set([]),
    'FEAT': set(['q']),
    'PART': set(['p', 'k', 'eps']),
    'OSMNX': set(['location_info']),
}


# ============== END ====================================== #

# ============== PyG converter params ===================== #


# ============== END ====================================== #