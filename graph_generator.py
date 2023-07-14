from params import SAMPLER_SPECS, GRAPH_TYPES
import numpy as np
import networkx as nx

_Array = np.ndarray


def _add_uniform_weights(adj, low, high):
    n, m = adj.shape
    weights = np.random.uniform(
        low=low, high=high, size=(n, m)
    )
    return np.multiply(adj.astype(float), weights)


def _sample_er_bipartite_graph(m: int, n: int, **kwargs):
    """Random Erdos-Renyi graph."""
    weighted = kwargs.get('weighted', False)
    low = kwargs.get('low', 0.0),
    high = kwargs.get('high', 1.0)
    p = kwargs.get('p', 0.5)

    mat = np.random.binomial(1, p, size=(m, n))
    if weighted:
        mat = _add_uniform_weights(mat, low, high)

    return mat


# TODO: Fix this implementation
def _sample_ba_bipartite_graph(m: int, n: int, **kwargs):
    weighted = kwargs.get('weighted', False)
    low = kwargs.get('low', 0.0),
    high = kwargs.get('high', 1.0)
    ba_param = kwargs.get('ba_param', 5)

    ba_graph = nx.barabasi_albert_graph(n + m, ba_param)
    mat = nx.to_numpy_array(ba_graph)[:n, n:]
    if weighted:
        mat = _add_uniform_weights(mat, low, high)

    return mat


def _sample_geom_bipartite_graph(m: int, n: int, **kwargs):
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

    red_rates = np.random.power(power, (partition, partition))
    blue_rates = np.random.power(power, (partition, partition))
    red_rates = (red_rates / np.sum(red_rates)).flatten()
    blue_rates = (blue_rates / np.sum(blue_rates)).flatten()

    bounds = [
        ((1 - (i + 1) * width, 1 - i * width),
         (1 - (j + 1) * width, 1 - j * width))
        for j in np.arange(partition)
        for i in np.arange(partition)
    ]

    red_ind = np.random.choice(
        np.arange(partition * partition), m, p=red_rates)
    blue_ind = np.random.choice(
        np.arange(partition * partition), n, p=blue_rates)

    red = []
    blue = []

    for i in red_ind:
        red.append(
            [np.random.uniform(*bounds[i][0]),
             np.random.uniform(*bounds[i][1])]
        )
    for i in blue_ind:
        blue.append(
            [np.random.uniform(*bounds[i][0]),
             np.random.uniform(*bounds[i][1])]
        )

    red = np.array(red)
    blue = np.array(blue)

    # m x n matrix with pairwise euclidean distances
    dist = np.linalg.norm(red[:, None, :] - blue[None, :, :], axis=-1)
    dist[dist < threshold] = 0
    return scaling * dist


def _sample_complete_bipartite_graph(m: int, n: int, **kwargs):
    return np.random.rand(m, n)


def sample_bipartite_graph(m: int, n: int, **kwargs) -> _Array:
    SAMPLER_ROUTER = {
        'ER': _sample_er_bipartite_graph,
        'BA': _sample_ba_bipartite_graph,
        'GEOM': _sample_geom_bipartite_graph,
        'COMP': _sample_complete_bipartite_graph
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

    return SAMPLER_ROUTER[graph_type](m, n, **kwargs)
