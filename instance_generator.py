from params import SAMPLER_SPECS, GRAPH_TYPES, _Array, _Instance
from util import _random_subset, _load_gmission
import numpy as np
import random
from numpy.random import Generator
from typing import List, Tuple


def _add_uniform_weights(
    adj: _Array,
    low: float,
    high: float,
    rng: Generator
) -> _Array:
    
    n, m = adj.shape
    weights = rng.uniform(
        low=low, high=high, size=(n, m)
    )
    return np.multiply(adj.astype(float), weights)


def _sample_er_bipartite_graph(m: int, n: int, rng: Generator, **kwargs):
    """Random Erdos-Renyi graph."""
    weighted = kwargs.get('weighted', False)
    low = kwargs.get('low', 0.0),
    high = kwargs.get('high', 1.0)
    p = kwargs.get('p', 0.5)

    mat = rng.binomial(1, p, size=(m, n))
    if weighted:
        mat = _add_uniform_weights(mat, low, high, rng)
    return mat


def _barabasi_albert_graph(
    m: int,
    n: int,
    ba_param: int,
    rng: Generator
) -> _Array:
    assert m >= ba_param, "Invalid specification"
    A = np.zeros((m, n))
    repeated_offline_nodes = list(range(n))
    for i in range(m):
        targets = _random_subset(repeated_offline_nodes, ba_param, rng)
        for j in targets:
            A[i, j] = 1
        repeated_offline_nodes.extend(targets)

    return A


def _sample_ba_bipartite_graph(m: int, n: int, rng:Generator, **kwargs):
    low = kwargs.get('low', 0.0),
    high = kwargs.get('high', 1.0)
    weighted = kwargs['weighted']
    ba_param = kwargs['ba_param']

    A = _barabasi_albert_graph(m, n, ba_param, rng)

    if weighted:
        A = _add_uniform_weights(A, low, high, rng)

    return A


def _sample_geom_bipartite_graph(m: int, n: int, rng: Generator, **kwargs):
    ''' Generates a geometric bipartite graph by embedding [n] LHS nodes and [m]
    RHS nodes uniformly in a unit square, and taking pairwise distances to be
    edge weights. Edges with weight below [threshold] are removed, and
    edge weights are scaled by a factor of [scaling] as a final step.
    '''

    threshold = kwargs.get('threshold', 0.25)
    scaling = kwargs.get('scaling', 1.0)
    partition = kwargs.get('partition', 5)
    width = 1 / partition
    power = kwargs.get('power', 1)

    red_rates = rng.power(power, (partition, partition))
    blue_rates = rng.power(power, (partition, partition))
    red_rates = (red_rates / np.sum(red_rates)).flatten()
    blue_rates = (blue_rates / np.sum(blue_rates)).flatten()

    bounds = [
        ((1 - (i + 1) * width, 1 - i * width),
         (1 - (j + 1) * width, 1 - j * width))
        for j in np.arange(partition)
        for i in np.arange(partition)
    ]

    red_ind = rng.choice(
        np.arange(partition * partition), m, p=red_rates)
    blue_ind = rng.choice(
        np.arange(partition * partition), n, p=blue_rates)

    red = []
    blue = []

    for i in red_ind:
        red.append(
            [rng.uniform(*bounds[i][0]),
             rng.uniform(*bounds[i][1])]
        )
    for i in blue_ind:
        blue.append(
            [rng.uniform(*bounds[i][0]),
             rng.uniform(*bounds[i][1])]
        )

    red = np.array(red)
    blue = np.array(blue)

    # m x n matrix with pairwise euclidean distances
    dist = np.linalg.norm(red[:, None, :] - blue[None, :, :], axis=-1)
    dist[dist < threshold] = 0

    return scaling * dist


def _sample_complete_bipartite_graph(m: int, n: int, rng: Generator, **kwargs):
    return rng.rand(m, n)


# TODO Fix random generation for GMission graphs
def _sample_gmission_bipartite_graph(m: int, n: int, rng: Generator, **kwargs):
    edge_df = _load_gmission()
    task_subset = random.sample(range(1, 712), m)
    worker_subset = random.sample(range(1, 533), n)

    subgraph_edges = edge_df[
        edge_df.worker_type.isin(worker_subset) &
        edge_df.task_type.isin(task_subset)
    ]

    A = np.array(
        subgraph_edges.pivot(
            index='task_type', columns='worker_type', values='weight')
        .fillna(value=0)
        .reindex(columns=worker_subset, index=task_subset, fill_value=0)
    )

    return A


def _location_feature(num_points: int, rng: Generator) -> _Array:
    return rng.uniform(0, 1, size=(num_points, 2))

def _rating_feature(num_points: int, rng: Generator) -> _Array:
    return rng.choice([0.2, 0.4, 0.6, 0.8, 1], size=(num_points, 1))

def _sample_synthetic_features(m: int, n: int, rng: Generator) -> _Array:
    loc_m = _location_feature(m, rng)
    rat_m = _rating_feature(m, rng)

    loc_n = _location_feature(n, rng)
    rat_n = _rating_feature(n, rng)

    return np.hstack([loc_m, rat_m]), np.hstack([loc_n, rat_n])


def _sample_feature_bipartite_graph(m: int, n: int, rng: Generator, **kwargs) -> _Array:
    M, N = _sample_synthetic_features(m, n, rng)
    q = kwargs.get('q', 0.5)
    score_matrix = M @ N.T
    threshold = np.quantile(score_matrix.flatten(), q)
    return (score_matrix >= threshold).astype(float)


def _sample_bipartite_graph(
    m: int,
    n: int,
    rng: Generator,
    **kwargs
) -> _Array:
    SAMPLER_ROUTER = {
        'ER': _sample_er_bipartite_graph,
        'BA': _sample_ba_bipartite_graph,
        'GEOM': _sample_geom_bipartite_graph,
        'COMP': _sample_complete_bipartite_graph,
        'GM': _sample_gmission_bipartite_graph,
        'FEAT': _sample_feature_bipartite_graph
    }

    graph_type = kwargs.get('graph_type', '[not provided]')

    if graph_type not in GRAPH_TYPES:
        raise ValueError(f'Invalid graph type: {graph_type}')

    provided_names = set(kwargs.keys())
    missing_names = SAMPLER_SPECS[graph_type].difference(provided_names)

    if len(missing_names) > 0:
        raise ValueError(
            'Did not provide required attributes for '
            f'{graph_type} graph type: {missing_names}'
        )
    
    return SAMPLER_ROUTER[graph_type](m, n, rng, **kwargs)


def _sample_bipartite_graphs(
    m: int,
    n: int,
    num: int,
    rng: Generator,
    **kwargs
) -> List[_Array]:
    
    return [
        _sample_bipartite_graph(m, n, rng, **kwargs)
        for _ in range(num)
    ]


def _sample_probs(
    m: int,
    num: int,
    rng: Generator
) -> _Array:
    
    # p = rng.uniform(0.5, 1, m)
    # return np.vstack([p for _ in range(num)]).T
    return rng.uniform(0.5, 1, (m, num))


def sample_instances(
    m: int,
    n: int,
    num: int,
    rng: Generator,
    **kwargs
) -> Tuple[List[_Instance], _Array]:
    
    As = _sample_bipartite_graphs(m, n, num, rng, **kwargs)
    ps = _sample_probs(m, num, rng)
    instances = [
        (As[i], ps[:, i])
        for i in range(len(As))
    ]
    return instances
