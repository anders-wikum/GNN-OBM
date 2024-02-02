import numpy as np

######## Training ########

META_TRAIN_CONFIG = {
    'train_num': 150,
    'val_num': 50,
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
	'regimes': [(30,10)]
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


EXPERIMENT_BASE_TESTING_REGIMES = [(30,10)]
EXPERIMENT_NOISE_ROBUSTNESS_REGIMES = [(30,10)]
EXPERIMENT_SIZE_GENERALIZATION_REGIMES = [(2*x, x) for x in np.arange(10, 40, 2)]
