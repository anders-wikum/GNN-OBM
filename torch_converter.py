from params import _Array, _Instance
from algorithms import one_step_stochastic_opt, cache_stochastic_opt
from util import diff, _neighbors
import numpy as np
import torch
from numpy.random import Generator
from torch_geometric.data import Data
from typing import List, Tuple
from copy import deepcopy

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


def _gen_edge_tensors(A: _Array) -> Tuple[torch.tensor, torch.tensor]:
    m, n = A.shape
    edge_index = []; edge_attr = []

    # Add edges/edge weights in underlying graph
    for i in range(m):
        for j in range(n):
            if A[i, j] > 0:
                edge_index.append([j, n + i])
                edge_index.append([n + i, j])
                edge_attr.append([A[i, j]])
                edge_attr.append([A[i, j]])
    
    # Add edges to virtual node representing no match
    for i in range(0, m):
        edge_index.append([n + m, n + i])
        edge_index.append([n + i, n + m])
        edge_attr.append([0])
        edge_attr.append([0])

    edge_index = torch.tensor(edge_index).T
    edge_attr = torch.tensor(edge_attr).type(torch.FloatTensor)
    return edge_index, edge_attr


def _gen_graph_features(m, n, offline_nodes, t):
    ratio = torch.tensor([(m - t) / len(offline_nodes)] * (n + m + 1))
    t = torch.tensor([t] * (n + m + 1))

    return torch.stack(
        [
            t,
            ratio,
        ]
    ).type(torch.FloatTensor)
    

def init_pyg(
    instance: _Instance,
    rng: Generator
) -> Data:
    '''
    Initializes a PyG data object representing the problem instance (A, p).
    '''
    A, p = instance
    m, n = A.shape

    offline_nodes = frozenset(np.arange(n))
    t = 0

    x = _gen_node_features(m, n, p, t, rng)
    edge_index, edge_attr = _gen_edge_tensors(A)
    neighbors = _neighbor_encoder(A, offline_nodes, t)
    graph_features = _gen_graph_features(m, n, offline_nodes, t)

    return Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        graph_features=graph_features,
        neighbors=neighbors,
        m=torch.tensor([m]),
        n=torch.tensor([n])
    )


def update_pyg(
    data: Data,
    instance: _Instance,
    choice: int,
    t: int,
    offline_nodes: frozenset
) -> None:
    A, _ = instance
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
        num_offline = len(offline_nodes)
        data.graph_features[0] = data.graph_features[0] + 1
        if num_offline == 0:
            data.graph_features[1] = torch.full(
                data.graph_features[1].size(),
                fill_value=-1
        )
        else:
            data.graph_features[1] = (m - t - 1) / num_offline

    _update_node_features()
    _update_edges()
    _update_graph_features()
    _update_neighbors()


def label_pyg(data: Data, y: _Array) -> Data:
    data.hint = torch.FloatTensor(y)
    return deepcopy(data)


def _density(instance: _Instance) -> float:
   A, _ = instance
   return (A > 0).sum() / (A >= 0).sum()


def _online_ratio(instance: _Instance) -> float:
    A, _ = instance
    m, n = A.shape
    return m / n


def _exp_clustering_coef_quantiles(
    instance: _Instance,
    threshold: float
) -> List[float]:
    A, p = instance
    return np.quantile(
        ((A > threshold).T @ p) / p.sum(),
        [0, 1]
    )


def _exp_graph_clustering_coef(
    instance: _Instance,
    threshold: float
) -> List[float]:
    A, p = instance
    return np.mean((A >= threshold).T @ p / p.sum())


def _edge_quantiles(instance: _Instance) -> List[float]:
    A, _ = instance
 
    return np.quantile(
        A[A > 0].flatten(),
        [0, 0.25, 0.5, 0.75, 1]
    )


def _expected_online_ratio(instance: _Instance) -> float:
    A, p = instance
    n = A.shape[1]
    return p.sum() / n


def _featurize(instance: _Instance) -> _Array:
    SCAL_FEAT_FNS = [
        (_online_ratio, [[]]),
        (_expected_online_ratio, [[]]),
        (_exp_graph_clustering_coef, [[0], [0.2], [0.4], [0.6], [0.8]]),
        (_density, [[]])
    ]
    VEC_FEAT_FNS = [
        (_edge_quantiles, [[]]),
        (_exp_clustering_coef_quantiles, [[0], [0.2], [0.4], [0.6], [0.8]])
    ]

    embedding = [
        *[
            func(instance, *kwargs)
            for (func, kwarg_list) in SCAL_FEAT_FNS
            for kwargs in kwarg_list
        ],
        *[
            el
            for (func, kwarg_list) in VEC_FEAT_FNS
            for kwargs in kwarg_list
            for el in func(instance, *kwargs)
        ]
    ]

    return np.array(embedding)


def init_features(instance: _Instance, rng: Generator) -> Data:
    return Data(X=_featurize(instance))


def update_features(
    data: Data,
    instance: _Instance,
    choice: int,
    t: int,
    offline_nodes: frozenset
) -> None:
    
    A, p = instance
    # Filter used offline nodes, online nodes that have arrived
    p_new = p[t:]
    A_new = A[t:, :]
    mask = [i in offline_nodes for i in np.arange(A.shape[1])]
    A_new = A_new[:, mask] 
    data = Data(X=_featurize((A_new, p_new)))


def label_features(data: Data, y: _Array) -> None:
    data.y = y


def _2d_normalize(arr: _Array):
    return (arr - arr.mean()) / arr.std()


def _marginal_vtg(instance, offline_nodes, t, cache, **kwargs):
    A, _ = instance
    hint = one_step_stochastic_opt(
        A, offline_nodes, t, cache
    )
    return hint - hint[-1]


def _skip_class(instance, offline_nodes, t, cache, **kwargs):
    A, _ = instance
    hint = one_step_stochastic_opt(
        A, offline_nodes, t, cache
    )
    return [1 * (np.argmax(hint) == len(hint) - 1)]


def _meta_ratios(instance, offline_nodes, t, cache, **kwargs):

    A, _ = instance
    hint = one_step_stochastic_opt(
        A, offline_nodes, t, cache
    )
    device = kwargs['device']
    data = kwargs['data']
    data.to(device)
    models = kwargs['models']

    choices = [
        torch.argmax(model(data)[data.neighbors]).item()
        for model in models
    ]

    data.to('cpu')
    vtgs = hint[choices] 
    
    if vtgs[0] > vtgs[1]:
        return [0]
    else:
        return [1]


LABEL_FUNCS = {
    'regression': _marginal_vtg,
    'classification': _skip_class,
    'meta': _meta_ratios
}

SAMPLE_INIT_FUNCS = {
    'gnn': init_pyg,
    'nn': init_features
}

SAMPLE_UPDATE_FUNCS = {
    'gnn': update_pyg,
    'nn': update_features
}

SAMPLE_LABEL_FUNCS = {
    'gnn': label_pyg,
    'nn': label_features
}

def _instance_to_sample_path(
    instance: _Instance,
    sample_type: str,
    label_fn: callable,
    cache: dict,
    rng: Generator,
    **kwargs
):
    A, _ = instance
    m, n = A.shape

    # Initialize data at start of sample path
    data = SAMPLE_INIT_FUNCS[sample_type](instance, rng)
    sample_path = []
    offline_nodes = frozenset(np.arange(n))
    kwargs['data'] = data
    
    for t in range(m):
        if len(offline_nodes) > 0:
            # Generate labels for current execution state
            labels = label_fn(
                instance,
                offline_nodes,
                t,
                cache,
                **kwargs
            )

            # Label data, add to sample path
            labeled_sample = SAMPLE_LABEL_FUNCS[sample_type](data, labels)
            sample_path.append(labeled_sample)

            # Update state / data
            choice = cache[t][offline_nodes][1]
            offline_nodes = diff(offline_nodes, choice)

            if t < m - 1:
                SAMPLE_UPDATE_FUNCS[sample_type](
                    data,
                    instance,
                    choice,
                    t + 1, 
                    offline_nodes
                )

    return sample_path

        
def _instances_to_gnn_train_samples(
    instances: List[_Instance],
    head: str,
    **kwargs
) -> list:
    
    gnn_samples = []
    rng = np.random.default_rng()
    label_fn = LABEL_FUNCS[head]

    for instance in instances:
        cache = cache_stochastic_opt(*instance)
        gnn_samples.extend(
            _instance_to_sample_path(
                instance,
                "gnn",
                label_fn,
                cache,
                rng,
                **kwargs
            )
        )
    return gnn_samples


def _instances_to_nn_samples(
        instances: List[_Instance],
        models: List[object],
        args: dict,
        batch_size: int,
        rng: Generator
    ):

    from evaluate import evaluate_model

    X = np.vstack([_featurize(instance) for instance in instances])
    labels = []
    for model in models:
        model_ratio, greedy_ratio = evaluate_model(
            classify_model=None,
            eval_models=[model],
            device=args['device'],
            instances=instances,
            batch_size=batch_size,
            rng=rng,
            num_realizations=10
        )
        labels.append(model_ratio)
    labels.append(greedy_ratio)
    labels = np.array(labels).T

    y = np.zeros(labels.shape)
    y[np.arange(y.shape[0]), np.argmax(labels, axis=1)] = 1

    return X, y


def _instances_to_nn_eval(instances: List[_Instance]):
    return torch.FloatTensor(
        np.vstack(
            [
                _featurize(instance)
                for instance in instances
            ]
        )
    )

