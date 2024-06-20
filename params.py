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

######## Graph configurations ########

TRAIN_CONFIGS = {
    'BASE': {
        'train_num': 2000,
        'val_num': 300,
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
        ]
    },
    'META': {
        'train_num': 2000,
        'val_num': 300,
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
        ]
    },
    'NOISE': {
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
        ]
    }
}

TEST_CONFIGS = {
    'SMALL': [
        {
            'graph_type': 'ER',
            'p': 0.5,
            'weighted': True
        },
        {
            'graph_type': 'OSMNX',
            'location': 'Fremont, California, USA'
        },
    ],
    'MAIN': [
        {
            'graph_type': 'ER',
            'p': 0.5,
            'weighted': True,
            'weight_scaling': 1
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
    ],
    'ALL': [
        {
            'graph_type': 'ER',
            'p': 0.25,
            'weighted': True
        },
        {
            'graph_type': 'ER',
            'p': 0.5,
            'weighted': True
        },
        {
            'graph_type': 'ER',
            'p': 0.75,
            'weighted': True
        },
        {
            'graph_type': 'BA',
            'ba_param': 4,
            'weighted': True
        },
        {
            'graph_type': 'BA',
            'ba_param': 6,
            'weighted': True
        },
        {
            'graph_type': 'BA',
            'ba_param': 8,
            'weighted': True
        },
        {
            'graph_type': 'GEOM',
            'q': 0.15,
            'd': 2
        },
        {
            'graph_type': 'GEOM',
            'q': 0.25,
            'd': 2
        },
        {
            'graph_type': 'GEOM',
            'q': 0.5,
            'd': 2
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
}

# ============== Regimes ================================== #

REGIMES = {
    'BASE_10_6_TRAIN': [(10, 6)],
    'BASE_6_10_TRAIN': [(6, 10)],
    'META_TRAIN': [(6, 10), (8, 8), (10, 6)],
    'BASE_TEST': [(30, 10)],
    'SIZE_GENERALIZATION': [(2*x, x) for x in np.arange(10, 95, 2)],
    'REGIME_GENERALIZATION': [(x, 16) for x in np.arange(8, 45, 3)]
}

# ============== END ====================================== #