from params import SAMPLER_SPECS, GRAPH_TYPES, _Array, _Instance
from util import _random_subset, _load_gmission, _load_osmnx
import numpy as np
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

    q = kwargs.get('q', 0.25)
    d = kwargs.get('d', 2)
    weighted = kwargs.get('weighted', True)

    red = rng.uniform(0, 1, (d, m))
    blue = rng.uniform(0, 1, (d, n))
    dist = np.linalg.norm(blue[:, None, :] - red[:, :, None], axis=0)
    
    #dist[dist > threshold] = 0
    dist = (np.max(dist) - dist) / np.max(dist)
    threshold = np.quantile(dist.flatten(), 1-q)
    dist[dist < threshold] = 0

    if not weighted:
        return(dist > 0).astype(float)
    
    dist[dist > 0] = dist[dist > 0] - np.min(dist[dist>0])
    dist /= np.max(dist)
    return dist


def _sample_complete_bipartite_graph(m: int, n: int, rng: Generator, **kwargs):
    return rng.rand(m, n)


def _sample_gmission_bipartite_graph(m: int, n: int, rng: Generator, **kwargs):
    edge_df = kwargs["data"]
    task_subset = rng.choice(range(1, 712), m, replace=False)
    worker_subset = rng.choice(range(1, 533), n, replace=False)

    subgraph_edges = edge_df[
        edge_df.worker_type.isin(worker_subset) &
        edge_df.task_type.isin(task_subset)
    ]
    A = np.array(
        subgraph_edges.pivot(
            index="task_type", columns="worker_type", values="weight")
        .fillna(value=0)
        .reindex(columns=worker_subset, index=task_subset, fill_value=0)
    )

    return A


def _sample_osmnx_graph(m: int, n: int, rng: Generator, **kwargs) -> _Array:
    # Give the location graph to avoid having to download it at each instance
    # The graph is expected to have travel times
    intersections = kwargs["intersections"]
    drive_times = kwargs["drive_times"]

    # Only keep matches if the driver can get to the user in less than 15 minutes
    threshold = kwargs.get('threshold', 15*60)
    travel_times = np.zeros((m, n))

    # Sample drivers and users from intersections in the topology graph
    riders = rng.choice(intersections, m)
    drivers = rng.choice(intersections, n)

    for i, rider in enumerate(riders):
        for j, driver in enumerate(drivers):
            try:
                travel_times[i, j] = drive_times[rider][driver]
            except:
                travel_times[i, j] = drive_times[driver].get(rider, 100000)

  # Only keep matches that could happen in a short enough amount of time
    travel_times[travel_times >= threshold] = 0
    matrix = (np.max(travel_times) - travel_times) / np.max(travel_times)
    return matrix


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
        'OSMNX': _sample_osmnx_graph
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
    
    batch_kwargs = _batch_kwargs(**kwargs)
    
    return SAMPLER_ROUTER[graph_type](m, n, rng, **batch_kwargs)


def _gmission_batch_kwargs(**kwargs) -> dict:
    kwargs['data'] = _load_gmission()
    return kwargs


def _osmnx_batch_kwargs(**kwargs) -> dict:
    location = kwargs['location']
    location_info = _load_osmnx(location)
    for key in location_info.keys():
        kwargs[key] = location_info[key]
    return kwargs


def _indentity_batch_kwargs(**kwargs) -> dict:
    return kwargs


def _batch_kwargs(**kwargs):
    KWARGS_ROUTER = {
        'ER': _indentity_batch_kwargs,
        'BA': _indentity_batch_kwargs,
        'GEOM': _indentity_batch_kwargs,
        'COMP': _indentity_batch_kwargs,
        'GM': _gmission_batch_kwargs,
        'OSMNX': _osmnx_batch_kwargs
    }

    graph_type = kwargs['graph_type']
    return KWARGS_ROUTER[graph_type](**kwargs)


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
    return rng.uniform(0, 1, (m, num))

def _add_noise_to_vector(array, rng, noise_std, clip = True):
    # Adds random gaussian noise with std noise_std. If clamp is True
    # the result gets clamped between 0 and 1 (used for probability distributions
    # and weights since they are always bounded 
    noisy_array = rng.normal(array, noise_std)
    if clip:
        noisy_array = np.clip(noisy_array, 0, 1)
    return noisy_array


def sample_instances(
    m: int,
    n: int,
    num: int,
    rng: Generator,
    args,
    **kwargs
) -> Tuple[List[_Instance], _Array]:
    noise_std = args.get('noise', 0)
    As = _sample_bipartite_graphs(m, n, num, rng, **kwargs)

    def _add_noise_if_nonzero(A, rng, noise_std):
        noisy_A = _add_noise_to_vector(A, rng, noise_std)
        noisy_A[A == 0] = 0
        return noisy_A

    noisy_As = [_add_noise_if_nonzero(A, rng, noise_std) for A in As]
    ps = _sample_probs(m, num, rng)
    noisy_ps = _add_noise_to_vector(ps, rng, noise_std)
    instances = [
        (As[i], ps[:, i], noisy_As[i], noisy_ps[:, i])
        for i in range(len(As))
    ]
    return instances
