import numpy as np
from typing import Tuple, List



# An instance is: A, p, noisy A, noisy p
_Array = np.ndarray
_Instance = Tuple[_Array, _Array, _Array, _Array]
_Matching = List[Tuple[int, int]]

# ============== Graph sampler params ===================== #

GRAPH_TYPES = ['GEOM', 'ER', 'BA', 'COMP', 'GM', 'FEAT', 'PART', 'OSMNX']

GROUP_SAMPLERS = ['GM', 'OSMNX']

SAMPLER_SPECS = {
    'GEOM': set(['q', 'd']),
    'ER': set(['p', 'weighted']),
    'BA': set(['ba_param', 'weighted']),
    'COMP': set([]),
    'GM': set([]),
    'FEAT': set(['q']),
    'PART': set(['p', 'k', 'eps']),
    'OSMNX': set(['location']),
}


# ============== END ====================================== #

# ================ Experiment params ====================== #
import numpy as np

######## Training ########

BASE_MODEL_TRAIN_CONFIG = {
	'train_num': 200,
	'val_num': 100,
	'configs': [
		{
			'graph_type': 'ER',
			'p': 0.75,
			'weighted': True
		},
		{
			'graph_type': 'BA',
			'ba_param': 2,
			'weighted': True
		},
		{
			'graph_type': 'GEOM',
			'q': 0.15,
			'd': 2,
			'weighted': True
		}
	],
	'regimes': [(10,6)]
}

META_TRAIN_CONFIG = {
    'train_num': 6,
    'val_num': 6,
    'configs': [
        {
            'graph_type': 'ER',
            'p': 1,
            'weighted': True
        },
        {
            'graph_type': 'BA',
            'ba_param': 4,
            'weighted': False
        },
        {
            'graph_type': 'GEOM',
            'q': 0.25,
            'd': 2,
            'weighted': True
        }
    ],
    'regimes': [(6, 10), (8, 8), (10, 6)]
}

NOISE_ROBUST_MODEL_TRAIN_CONFIG = {
	'train_num': 200,
	'val_num': 100,
	'configs': [
		{
			'graph_type': 'ER',
			'p': 0.75,
			'weighted': True
		},
		{
			'graph_type': 'BA',
			'ba_param': 2,
			'weighted': True
		},
		{
			'graph_type': 'GEOM',
			'q': 0.15,
			'd': 2,
			'weighted': True
		}
	],
	'regimes': [(30,10)]
}

THRESHOLD_GREEDY_NODE_REGIMES = [(30,10)]

######## TESTING ########

# All the graph testing configurations
ALL_TESTING_GRAPH_CONFIGS = [
    {
        'graph_type': 'ER',
        'p': 0.25,
        'weighted': True,
        'weight_scaling': 1
    },
    {
        'graph_type': 'ER',
        'p': 0.5,
        'weighted': True,
        'weight_scaling': 1
    },
    {
        'graph_type': 'ER',
        'p': 0.75,
        'weighted': True,
        'weight_scaling': 1
    },
    {
        'graph_type': 'BA',
        'ba_param': 4,
        'weighted': True,
        'weight_scaling': 1
    },
    {
        'graph_type': 'BA',
        'ba_param': 6,
        'weighted': True,
        'weight_scaling': 1
    },
    {
        'graph_type': 'BA',
        'ba_param': 8,
        'weighted': True,
        'weight_scaling': 1
    },
    {
        'graph_type': 'GEOM',
        'q': 0.15,
        'd': 2,
        'weighted': True
    },
     {
        'graph_type': 'GEOM',
        'q': 0.25,
        'd': 2,
        'weighted': True
    },
    {
        'graph_type': 'GEOM',
        'q': 0.5,
        'd': 2,
        'weighted': True
    },
    {
        'graph_type': 'OSMNX',
        'location': 'Piedmont, California, USA'
    },
    {
        'graph_type': 'OSMNX',
        'location': 'Fremont, California, USA'
    },
    {
        'graph_type': 'GM'
    }
]

# A subset of the graph configurations used for the main results
MAIN_TESTING_GRAPH_CONFIGS = [
    {
        'graph_type': 'ER',
        'p': 0.5,
        'weighted': True
    },
    {
        'graph_type': 'GEOM',
        'q': 0.25,
        'd': 2,
        'weighted': True
    },
    {
        'graph_type': 'OSMNX',
        'location': 'Fremont, California, USA'
    },
    {
        'graph_type': 'GM'
    }
]

EXPERIMENT_META_REGIMES = [(x, 16) for x in np.arange(8, 65, 4)]
EXPERIMENT_BASE_TESTING_REGIMES = [(30,10)]
EXPERIMENT_NOISE_ROBUSTNESS_REGIMES = [(30,10)]
EXPERIMENT_SIZE_GENERALIZATION_REGIMES = [(2*x, x) for x in np.arange(10, 200, 2)]


# ============== END ====================================== #