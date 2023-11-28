import numpy as np
from typing import Tuple

_Array = np.ndarray
_Instance = Tuple[_Array, _Array]

# ============== Graph sampler params ===================== #

GRAPH_TYPES = ['GEOM', 'ER', 'BA', 'COMP', 'GM', 'FEAT', 'PART']

SAMPLER_SPECS = {
    'GEOM': set(['threshold', 'scaling']),
    'ER': set(['p', 'weighted']),
    'BA': set(['ba_param', 'weighted']),
    'COMP': set([]),
    'GM': set([]),
    'FEAT': set(['q']),
    'PART': set(['p', 'size', 'eps'])
}

# ============== END ====================================== #

# ============== PyG converter params ===================== #


# ============== END ====================================== #