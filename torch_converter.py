from params import _Array, _Instance
from algorithms import one_step_stochastic_opt, cache_stochastic_opt
from util import _extract_edges, diff, _neighbors, fill_list
from typing import Optional
import numpy as np
import torch
from numpy.random import Generator
from torch_geometric.data import Data
from typing import Optional, Tuple, List


def _gen_mask(size: tuple, slice: object) -> _Array:
    mask = torch.zeros(size)
    mask[slice] = 1
    return mask


def _positional_encoder(size: int, rng: Generator):
    return torch.tensor(rng.uniform(0, 1, size))


def _arrival_encoder(p: _Array, t: int, size: int):
    p = np.copy(p)
    p[t] = 1
    p[:t] = 1
    fill_size = size - len(p) - 1
    return torch.tensor([*([0] * fill_size), *p, 0])


def _neighbor_encoder(A, offline_nodes, t):
    m, n = A.shape
    N_t = _neighbors(A, offline_nodes, t)
    return torch.tensor([*[u in N_t for u in np.arange(n + m)], True])


def _gen_node_features(m: int, n: int, p: _Array, t: int, rng: Generator):
    offline_mask = _gen_mask(n + m + 1, slice(0, n, 1))
    arrival_mask = _gen_mask(n + m + 1, n + t)
    pos_encoder = _positional_encoder(n + m + 1, rng)
    arrival_probs = _arrival_encoder(p, t, n + m + 1)

    return torch.stack(
        [
            pos_encoder,
            offline_mask,
            arrival_probs,
            arrival_mask
        ]
    ).T.type(torch.FloatTensor)


def _gen_edge_tensors(A, offline_nodes, t):
    edge_index, edge_attr = _extract_edges(A, offline_nodes, t)
    edge_index = torch.tensor(edge_index).T
    edge_attr = torch.tensor(edge_attr).type(torch.FloatTensor)
    return edge_index, edge_attr


def _gen_graph_features(m, n, offline_nodes, t):
    ratio = torch.tensor([len(offline_nodes) / (m - t)] * (n + m + 1))
    t = torch.tensor([t] * (n + m + 1))

    return torch.stack(
        [
            t,
            ratio,
        ]
    ).type(torch.FloatTensor)
    

def _to_pyg(
    A: _Array,
    p: _Array,
    offline_nodes: frozenset,
    t: int,
    rng: Generator,
    hint: Optional[_Array] = None
):
    '''
    Generates a data sample. [A] is the adjacency matrix consisting of
    unmatched offline nodes and online nodes which have not already been
    seen. For each of these online nodes, [p] gives the node's arrival
    probability (first entry is always 1 and corresponds to arriving
    online node.

    '''
    m, n = A.shape
    
    x = _gen_node_features(m, n, p, t, rng)
    edge_index, edge_attr = _gen_edge_tensors(A, offline_nodes, t)
    neighbors = _neighbor_encoder(A, offline_nodes, t)
    graph_features = _gen_graph_features(m, n, offline_nodes, t)

    instance = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        graph_features=graph_features,
        neighbors=neighbors,
    )

    if hint is not None:
        instance.hint = torch.FloatTensor(hint)

    return instance


def _update_pyg(data: Data, t: int, choice: int, A: _Array, offline_nodes: frozenset):
    m, n = A.shape
    def _filter_choice_edges(edge_index, edge_attr):
        mask = ((edge_index != choice) & (edge_index != n + t - 1)).all(dim=0)
        return edge_index[:, mask], edge_attr[mask, :]
    
    def _update_node_features():
        data.x[n + t, 2] = 1
        data.x[n + t, 3] = 1
        data.x[n + t - 1, 3] = 0

    def _update_edges():
        edge_index, edge_attr = _filter_choice_edges(
            data.edge_index,
            data.edge_attr
        )
        data.edge_index = edge_index
        data.edge_attr = edge_attr

    def _update_neighbors():
        data.neighbors = _neighbor_encoder(A, offline_nodes, t)
    
    def _update_graph_features():
        data.graph_features[0] = data.graph_features[0] + 1
        data.graph_features[1] = data.graph_features[1] * (m - t) / (m - t + 1)

    _update_node_features()
    _update_edges()
    _update_graph_features()
    _update_neighbors()


def _marginal_vtg(A, p, offline_nodes, t, cache, rng, **kwargs):
    hint = one_step_stochastic_opt(
        A, offline_nodes, t, cache
    )
    hint = hint - hint[-1]
    return _to_pyg(A, p, offline_nodes, t, rng, hint)


def _skip_class(A, p, offline_nodes, t, cache, rng, **kwargs):
    hint = one_step_stochastic_opt(
        A, offline_nodes, t, cache
    )
    hint = [1 * (np.argmax(hint) == len(hint) - 1)]
    return _to_pyg(A, p, offline_nodes, t, rng, hint)


INSTANCE_GEN_FUNCS = {
    'regression': _marginal_vtg,
    'classification': _skip_class
}


def _instance_to_gnn_sample_path(
    instance: _Instance,
    head: str,
    cache: dict,
    rng: Generator,
    **kwargs: dict
):
    sample_path = []
    A, p = instance
    m, n = A.shape

    offline_nodes = frozenset(np.arange(n))
    matched_nodes = set()

    for t in range(m):
        graph = INSTANCE_GEN_FUNCS[head](
            A,
            p,
            offline_nodes,
            t,
            cache,
            rng,
            **kwargs 
        )
        sample_path.append(graph)
        choice = cache[t][offline_nodes][1]
        if choice != -1:
            matched_nodes.add(choice)
            matched_nodes.add(n + t)

        offline_nodes = diff(offline_nodes, choice)

    return sample_path


def _instances_to_gnn_train_samples(
    instances: List[_Instance],
    head: str
) -> list:
    
    gnn_samples = []
    rng = np.random.default_rng()

    for instance in instances:
        cache = cache_stochastic_opt(*instance)
        gnn_samples.extend(
            _instance_to_gnn_sample_path(instance, head, cache, rng)
        )
    return gnn_samples


def _instances_to_gnn_eval_state(
    instances: List[_Instance],
    coin_flips: _Array,
    rng: Generator
):
    num_instances = len(instances)
    _, n = instances[0][0].shape
    init_offline = frozenset(np.arange(n))

    state = {
        'offline_nodes': fill_list(init_offline, num_instances),
        'matchings': fill_list([], num_instances),
        'values':  fill_list(0, num_instances),
        'As': [A for (A, _) in instances],
        'coin_flips': coin_flips,
        'dataset': [
            _to_pyg(A, p, init_offline, 0, rng)
            for (A, p) in instances
        ]
    }

    return state


def _density(instance: _Instance) -> float:
   A, _ = instance
   return (A > 0).sum() / (A >= 0).sum()

def _online_ratio(instance: _Instance) -> float:
    A, _ = instance
    m, n = A.shape
    return m / n


def _degree_quantiles(instance: _Instance) -> List[float]:
    A, _ = instance
    return np.quantile(
        (A > 0).sum(axis=0) / A.shape[0],
        [0, 0.25, 0.5, 0.75, 1]
    )


def _edge_quantiles(instance: _Instance) -> List[float]:
    A, _ = instance
    return np.quantile(
        A[A > 0].flatten(),
        [0, 0.25, 0.5, 0.75, 1]
    )

def _prob_quantiles(instance: _Instance) -> List[float]:
    _, p = instance
    return np.quantile(
        p,
        [0, 0.25, 0.5, 0.75, 1]
    )


def _featurize(instance: _Instance) -> _Array:
    SCAL_FEAT_FNS = [_online_ratio, _density]
    VEC_FEAT_FNS = [_edge_quantiles, _degree_quantiles, _prob_quantiles]

    embedding = [
        *[func(instance) for func in SCAL_FEAT_FNS],
        *[el for func in VEC_FEAT_FNS for el in func(instance)]
    ]

    return np.array(embedding)

def _2d_normalize(arr: _Array):
    return (arr - arr.mean()) / arr.std()


def _instances_to_nn_samples(
        instances: List[_Instance],
        coin_flips: _Array,
        models: List[object],
        args: dict,
        batch_size: int,
        rng: Generator
    ):

    from evaluate import evaluate_model

    X = np.vstack([_featurize(instance) for instance in instances])
    y = []
    for model in models:
        (model_ratio, greedy_ratio), _ = evaluate_model(
            model,
            args,
            instances,
            coin_flips,
            batch_size,
            rng
        )
        y.append(model_ratio)
    y.append(greedy_ratio)
    y = np.array(y).T
    return X, _2d_normalize(y)


def _instances_to_nn_eval(instances: List[_Instance]):
    return torch.FloatTensor(
        np.vstack(
            [
                _featurize(instance)
                for instance in instances
            ]
        )
    )

