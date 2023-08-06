from params import SAMPLER_SPECS, GRAPH_TYPES, _Array
from obm_dp import one_step_stochastic_opt, cache_stochastic_opt
from util import _random_subset, _extract_edges, diff, _neighbors, _load_gmission
import pandas as pd
import numpy as np
import networkx as nx
import torch
import random
from torch_geometric.data import Data


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


def _add_edges_to_graph(
    G,
    source: list,
    targets: list,
    ba_param: int,
    source_repeated_nodes: list,
    target_repeated_nodes: list,
) -> None:

    G.add_edges_from(zip([source] * ba_param, targets))
    target_repeated_nodes.extend(targets)
    source_repeated_nodes.extend([source] * ba_param)


def _barabasi_albert_graph(m: int, n: int, ba_param: int) -> _Array:
    # Add m initial nodes (m0 in barabasi-speak)
    G = nx.empty_graph(2 * ba_param)
    G.name = "barabasi_albert_graph(%s, %s, %s)" % (m, n, ba_param)

    # List of existing nodes, with nodes repeated once for each adjacent edge
    LHS_repeated_nodes = list(range(0, ba_param))
    RHS_repeated_nodes = list(range(m, m + ba_param))

    # Start adding the other n-m nodes. The first node is m.
    LHS_source = 0
    RHS_source = m

    alternate = True
    next_node = 'LHS'

    while LHS_source < m or RHS_source < m + n:
        if RHS_source == n + m:
            next_node = 'LHS'
            alternate = False
        if LHS_source == m:
            next_node = 'RHS'
            alternate = False

        if next_node == 'LHS':
            RHS_targets = _random_subset(RHS_repeated_nodes, ba_param)
            _add_edges_to_graph(G, LHS_source, RHS_targets,
                                ba_param, LHS_repeated_nodes, RHS_repeated_nodes)
            LHS_source += 1
            if alternate:
                next_node = 'RHS'

        else:
            LHS_targets = _random_subset(LHS_repeated_nodes, ba_param)
            _add_edges_to_graph(G, RHS_source, LHS_targets,
                                ba_param, RHS_repeated_nodes, LHS_repeated_nodes)
            G.add_edges_from(zip([RHS_source] * ba_param, LHS_targets))
            RHS_source += 1
            if alternate:
                next_node = 'LHS'

    return G


def _sample_ba_bipartite_graph(m: int, n: int, **kwargs):
    low = kwargs.get('low', 0.0),
    high = kwargs.get('high', 1.0)
    weighted = kwargs['weighted']
    ba_param = kwargs['ba_param']

    ba_graph = _barabasi_albert_graph(m, n, ba_param)
    A = np.zeros((m, n))
    for (u, v) in ba_graph.edges():
        if u > v:
            A[v, u - m] = 1
        else:
            A[u, v - m] = 1

    if weighted:
        A = _add_uniform_weights(A, low, high)

    return A


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


def _sample_gmission_bipartite_graph(m: int, n: int, **kwargs):
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


def sample_bipartite_graph(m: int, n: int, **kwargs) -> _Array:
    SAMPLER_ROUTER = {
        'ER': _sample_er_bipartite_graph,
        'BA': _sample_ba_bipartite_graph,
        'GEOM': _sample_geom_bipartite_graph,
        'COMP': _sample_complete_bipartite_graph,
        'GM': _sample_gmission_bipartite_graph
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


def _gen_mask(size: tuple, slice: object) -> _Array:
    mask = torch.zeros(size)
    mask[slice] = 1
    return mask


def _positional_encoder(size: int):
    return torch.tensor(np.random.uniform(0, 1, size))


def _arrival_encoder(p: _Array, t: int, size: int):
    p = np.copy(p)
    p[t] = 1
    fill_size = size - len(p) - 1
    return torch.tensor([*([0] * fill_size), *p, 0])


def _to_pyg_train(
    A: _Array,
    p: _Array,
    offline_nodes: frozenset,
    t: int,
    hint: _Array
):
    x, edge_index, edge_attr, neighbor_mask, graph_features = _to_pyg(
        A, p, offline_nodes, t)

    hint = torch.tensor(hint).type(torch.FloatTensor)

    return Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        neighbors=neighbor_mask,
        hint=hint,
        graph_features=graph_features
    )


def _to_pyg_test(
    A: _Array,
    p: _Array,
    offline_nodes: frozenset,
    t: int

):
    x, edge_index, edge_attr, neighbor_mask, graph_features = _to_pyg(
        A, p, offline_nodes, t)
    return Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        neighbors=neighbor_mask,
        graph_features=graph_features
    )


def _neighbor_encoder(A, offline_nodes, t):
    m, n = A.shape
    N_t = _neighbors(A, offline_nodes, t)
    return torch.tensor([*[u in N_t for u in np.arange(n + m)], True])


def _to_pyg(
    A: _Array,
    p: _Array,
    offline_nodes: frozenset,
    t: int
):
    '''
    Generates a data sample. [A] is the adjacency matrix consisting of
    unmatched offline nodes and online nodes which have not already been
    seen. For each of these online nodes, [p] gives the node's arrival
    probability (first entry is always 1 and corresponds to arriving
    online node.

    '''
    m, n = A.shape
    edge_index, edge_attr = _extract_edges(A, offline_nodes, t)

    offline_mask = _gen_mask(n + m + 1, slice(0, n, 1))
    arrival_mask = _gen_mask(n + m + 1, n + t)
    pos_encoder = _positional_encoder(n + m + 1)
    neighbor_mask = _neighbor_encoder(A, offline_nodes, t)
    arrival_probs = _arrival_encoder(p, t, n + m + 1)

    ratio = torch.tensor([len(offline_nodes) / (m - t)] * (n + m + 1))
    t = torch.tensor([t] * (n + m + 1))

    graph_features = torch.stack(
        [
            t,
            ratio,
        ]).type(torch.FloatTensor)

    x = torch.stack(
        [
            pos_encoder,
            offline_mask,
            arrival_probs,
            arrival_mask
        ]).T.type(torch.FloatTensor)
    edge_index = torch.tensor(edge_index).T
    edge_attr = torch.tensor(edge_attr).type(torch.FloatTensor)
    # edge_attr = torch.stack(
    #     [
    #         torch.tensor(edge_attr),
    #         torch.tensor(edge_probs)
    #     ]).T.type(torch.FloatTensor).squeeze()

    return x, edge_index, edge_attr, neighbor_mask, graph_features


def _generate_example(A: _Array, ps: _Array, head: str, cache: dict):
    graphs = []

    ps = list(np.copy(ps))
    m, n = A.shape
    offline_nodes = frozenset(np.arange(n))
    matched_nodes = set()

    for t in range(m):

        hint = one_step_stochastic_opt(
            A, offline_nodes, t, cache)
        if head == 'regression':
            hint = hint - hint[-1]
        elif head == 'classification':
            hint = [1 * (np.argmax(hint) == len(hint) - 1)]
        else:
            raise NotImplemented

        graphs.append(_to_pyg_train(A, ps, offline_nodes, t, hint))
        choice = cache[t][offline_nodes][1]
        if choice != -1:
            matched_nodes.add(choice)
            matched_nodes.add(n + t)

        offline_nodes = diff(offline_nodes, choice)

    return graphs


def generate_examples(num: int, m: int, n: int, ps: _Array, head: str, **kwargs):
    dataset = []
    for _ in range(num):
        A = sample_bipartite_graph(m, n, **kwargs)
        cache = cache_stochastic_opt(A, ps)
        dataset.extend(_generate_example(A, ps, head, cache))
    return dataset
