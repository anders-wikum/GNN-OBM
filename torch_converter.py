from params import _Array, _Instance
from algorithms import one_step_stochastic_opt, cache_stochastic_opt
from util import diff, _neighbors
import numpy as np
import torch
from numpy.random import Generator
from torch_geometric.data import Data
from typing import List, Tuple, Optional
from copy import deepcopy
import time

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
    pos_encoder = _positional_encoder(n + m + 1, rng)
    offline_mask = _gen_mask(n + m + 1, slice(0, n, 1))
    arrival_probs = _arrival_encoder(p, t, n + m + 1)
    arrival_mask = _gen_mask(n + m + 1, n + t)
    skip_node_encoder = _gen_mask(n + m + 1, n + m)

    return torch.stack(
        [
            pos_encoder,
            offline_mask,
            arrival_probs,
            arrival_mask,
            skip_node_encoder
        ]
    ).T.type(torch.FloatTensor)



def _gen_edge_features(A: _Array, A_values: _Array) -> Tuple[torch.tensor, torch.tensor]:
    # A is used to decide if an edge is added, A_values is used
    # to add the values to edge_attr. This is useful to use noisy
    # weight values 
    m, n = A.shape
    edge_index = []; edge_attr = []

    # Add edges/edge weights in underlying graph
    for i in range(m):
        for j in range(n):
            if A[i, j] > 0:
                edge_index.append([j, n + i])
                edge_index.append([n + i, j])
                edge_attr.append([A_values[i, j]])
                edge_attr.append([A_values[i, j]])
    
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
    # ratio = torch.tensor([(m - t) / len(offline_nodes)] * (n + m + 1))
    # t = torch.tensor([t] * (n + m + 1))

    ratio = torch.Tensor([(m - t) / len(offline_nodes)])
    t = torch.Tensor([t])

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
    Initializes a PyG data object representing the problem instance (A, p, noisy_A, noisy_p).
    '''
    A, _, noisy_A, noisy_p = instance
    m, n = A.shape

    offline_nodes = frozenset(np.arange(n))
    t = 0

    x = _gen_node_features(m, n, noisy_p, t, rng)
    edge_index, edge_attr = _gen_edge_features(A, noisy_A)

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
    
    A, _, _, _ = instance
    m, n = A.shape

    def _filter_choice_edges(edge_index, edge_attr):
        mask = ((edge_index != choice) & (edge_index != n + t - 1)).all(dim=0)
        return edge_index[:, mask], edge_attr[mask, :]
    
    def _update_node_features():
        # Index 2 => arrival probabilities
        # Index 3 => arrival mask
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

    node_start = time.perf_counter()
    _update_node_features()
    node_end = time.perf_counter()
    edge_start = time.perf_counter()
    _update_edges()
    edge_end = time.perf_counter()
    graph_start = time.perf_counter()
    _update_graph_features()
    graph_end = time.perf_counter()
    neighbor_start = time.perf_counter()
    _update_neighbors()
    neighbor_end = time.perf_counter()
    # print(f"Node update: {node_end - node_start}")
    # print(f"Edge update: {edge_end - edge_start}")
    # print(f"Graph update: {graph_end - graph_start}")
    # print(f"Neighbor update: {neighbor_end - neighbor_start}")


def label_pyg(data: Data, y: _Array) -> Data:
    data.hint = torch.FloatTensor(y)
    return deepcopy(data)


def _marginal_vtg(instance, offline_nodes, t, cache, **kwargs):
    A, _, _, _ = instance
    hint = one_step_stochastic_opt(
        A, offline_nodes, t, cache
    )

    return hint - hint[-1]


def _skip_class(instance, offline_nodes, t, cache, **kwargs):
    A, _, _, _ = instance
    # Return the actual value to go. It is turned into a discrete
    # label when computing loss/accuracy (it makes passing the values 
    # around easier for batching)
    hint = one_step_stochastic_opt(
        A, offline_nodes, t, cache
    )

    # return [1 * (np.argmax(hint) == len(hint) - 1)]
    return hint


def _meta_ratios(instance, offline_nodes, t, cache, **kwargs):
    A, _, _, _ = instance
    hint = one_step_stochastic_opt(
        A, offline_nodes, t, cache
    )
    models = kwargs['models']
    data = kwargs['data']

    choices = [
        model.batch_select_match_nodes([data]).item()
        for model in models
    ]

    index_of_choice = {}
    neighbor_index = 0
    for i, neighbor in enumerate(data.neighbors):
        if neighbor:
            index_of_choice[i] = neighbor_index
            neighbor_index += 1

    neighor_indices = [
        index_of_choice.get(choice, len(hint) - 1)
        for choice in choices
    ]

    data.to('cpu')
    vtgs = hint[neighor_indices]
    label = np.zeros(len(models))

    all_indices = np.arange(len(vtgs))
    max_indices = all_indices[vtgs == np.max(vtgs)]
    if len(max_indices) == 1:
        label[max_indices[0]] = 1
    
    return label

    
TARGET_FUNCS = {
    'regression': _marginal_vtg,
    'classification': _skip_class,
    'meta': _meta_ratios
}


def _instance_to_sample_path(
    instance: _Instance,
    target_fn: callable,
    cache: dict,
    rng: Generator,
    base_models: List[object] | None
):
    A, _, _, _ = instance
    m, n = A.shape

    # Initialize data at start of sample path
    # There is a disctinction between the pyg graph for the base models and the
    # pyg graph for the meta model, the latter includes base model predictions as
    # a node feature
    pyg_graph = init_pyg(instance, rng)

    sample_path = []
    offline_nodes = frozenset(np.arange(n))
    kwargs = {'data': pyg_graph, 'models': base_models}
    
    for t in range(m):
        if len(offline_nodes) > 0:
            # Generate labels for current execution state
            targets = target_fn(
                instance,
                offline_nodes,
                t,
                cache,
                **kwargs
            )

            if not np.all(targets == 0):
                # Label data, add to sample path
                labeled_sample = label_pyg(pyg_graph, targets)
                sample_path.append(labeled_sample)

       
            # Update state / data
            choice = cache[t][offline_nodes][1]
            offline_nodes = diff(offline_nodes, choice)

            if t < m - 1 and len(offline_nodes) > 0:
                # Update the pyg graph for the base models
                update_pyg(
                    pyg_graph,
                    instance,
                    choice,
                    t + 1,
                    offline_nodes
                )

    return sample_path

        
def _instances_to_train_samples(
    instances: List[_Instance],
    head: str,
    base_models: Optional[List[object]] = None,
) -> list:
    
    samples = []
    rng = np.random.default_rng()
    target_fn = TARGET_FUNCS[head]

    for instance in instances:
        cache = cache_stochastic_opt(instance[0], instance[1])
        samples.extend(
            _instance_to_sample_path(
                instance,
                target_fn,
                cache,
                rng,
                base_models
            )
        )
    return samples
