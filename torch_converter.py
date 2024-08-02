import torch
import numpy as np
from copy import deepcopy
from numpy.random import Generator
from torch_geometric.data import Data
from typing import Optional

from algorithms import one_step_stochastic_opt, cache_stochastic_opt
from params import Array, Tensor
from util import diff, _neighbors
from instance import Instance


def _mask(size: tuple, slice: object) -> Tensor:
    mask = torch.zeros(size)
    mask[slice] = 1
    return torch.tensor(mask)


def _positional_encoder(size: int, rng: Generator) -> Tensor:
    return torch.tensor(rng.uniform(0, 1, size))


def _arrival_encoder(instance: Instance, t: int, size: int) -> Tensor:
    p = np.copy(instance.p)
    p[t] = 1
    p[:t] = 1
    fill_size = size - len(p) - 1
    return torch.tensor([*([0] * fill_size), *p, 0])


def _neighbor_encoder(instance, offline_nodes, t) -> Tensor:
    m = instance.m
    n = instance.n
    A = instance.A
    N_t = _neighbors(A, offline_nodes, t)
    return torch.tensor([*[u in N_t for u in np.arange(n + m)], True])


def _gen_node_features(
    instance: Instance,
    t: int,
    rng: Generator
) -> torch.Tensor:
    
    m = instance.m
    n = instance.n
    p = instance.p

    pos_encoder = _positional_encoder(n + m + 1, rng)
    offline_mask = _mask(n + m + 1, slice(0, n, 1))
    arrival_probs = _arrival_encoder(p, t, n + m + 1)
    arrival_mask = _mask(n + m + 1, n + t)
    skip_node_encoder = _mask(n + m + 1, n + m)

    return torch.stack(
        [
            pos_encoder,
            offline_mask,
            arrival_probs,
            arrival_mask,
            skip_node_encoder
        ]
    ).T.type(torch.FloatTensor)


def _gen_edge_features(instance: Instance) -> tuple[Tensor, Tensor]:
    A = instance.A
    m = instance.m
    n = instance.n

    edge_index = []; edge_attr = []

    # Add edges/edge weights in underlying graph
    for i in range(m):
        for j in range(n):
            if A[i, j] > 0:
                edge_index.extend([[j, n + i], [n + i, j]])
                edge_attr.extend([[A[i, j]], [A[i, j]]])
    
    # Add edges to virtual node representing no match
    for i in range(0, m):
        edge_index.extend([[n + m, n + i], [n + i, n + m]])
        edge_attr.extend([[0], [0]])

    edge_index = torch.tensor(edge_index).T
    edge_attr = torch.tensor(edge_attr).type(torch.FloatTensor)
    return edge_index, edge_attr


def _gen_graph_features(
    instance: Instance,
    offline_nodes: frozenset,
    t: int
) -> torch.Tensor:
    
    m = instance.m
    ratio = torch.Tensor([(m - t) / len(offline_nodes)])
    t = torch.Tensor([t])

    return torch.stack(
        [
            t,
            ratio,
        ]
    ).type(torch.FloatTensor)
    

def init_pyg(
    instance: Instance,
    rng: Generator
) -> Data:
    '''
    Initializes a PyG data object representing the problem instance (A, p, noisy_A, noisy_p).
    '''

    all_nodes = frozenset(np.arange(instance.n))
    x = _gen_node_features(instance, t=0, rng=rng)
    edge_index, edge_attr = _gen_edge_features(instance)
    neighbors = _neighbor_encoder(instance, offline_nodes=all_nodes, t=0)
    graph_features = _gen_graph_features(instance, offline_nodes=all_nodes, t=0)

    return Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        graph_features=graph_features,
        neighbors=neighbors,
        m=torch.tensor([instance.m]),
        n=torch.tensor([instance.n])
    )


def update_pyg_with_choice(
    data: Data,
    instance: Instance,
    choice: int,
    t: int,
    offline_nodes: frozenset
) -> None:
    
    m, n = instance.m, instance.n

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
        data.neighbors = _neighbor_encoder(instance.A, offline_nodes, t)
    
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



def label_pyg(data: Data, y: Array) -> Data:
    data.hint = torch.FloatTensor(y)
    return deepcopy(data)


def _marginal_vtg(
    instance: Instance,
    offline_nodes: frozenset, 
    t: int,
    cache: dict,
    **kwargs
) -> Array:
    
    hint = one_step_stochastic_opt(instance.A, offline_nodes, t, cache)
    return hint - hint[-1]


def _meta_ratios(
    instance: Instance,
    offline_nodes: frozenset,
    t: int,
    cache: dict, 
    **kwargs
) -> Array:
    
    hint = one_step_stochastic_opt(instance.A, offline_nodes, t, cache)
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
    'meta': _meta_ratios
}


def _get_base_predictions(
    batch: Data,
    base_models: list[torch.nn.Module]
):
    device = next(base_models[0].parameters()).device
    batch.to(device)
    with torch.no_grad():
        base_pred = torch.concat(
            [
                base_model(batch)
                for base_model in base_models
            ],
            dim=1
        )
        return base_pred

def _instance_to_sample_path(
    instance: Instance,
    target_fn: callable,
    cache: dict,
    rng: Generator,
    base_models: list[object] | None
) -> list[Data]:
    
    m = instance.m
    n = instance.n
    pyg_graph = init_pyg(instance, rng)

    if base_models:
        pyg_graph.base_model_preds = _get_base_predictions(pyg_graph, base_models)

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
                update_pyg_with_choice(
                    pyg_graph,
                    instance,
                    choice,
                    t + 1,
                    offline_nodes
                )
                if base_models:
                    pyg_graph.base_model_preds = _get_base_predictions(pyg_graph, base_models)

    return sample_path

        
def _instances_to_train_samples(
    instances: list[Instance],
    head: str,
    base_models: Optional[list[object]] = None,
) -> list[Data]:
    
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
