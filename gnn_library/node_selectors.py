from abc import ABC, abstractmethod
import torch
import math
from util import Dataset

def _masked_argmax(tensor, mask, dim):
    masked = torch.mul(tensor, mask)
    neg_inf = torch.zeros_like(tensor)
    neg_inf[~mask] = -math.inf 
    return (masked + neg_inf).argmax(dim=dim)


def _vtg_greedy_choices(pred: torch.Tensor, batch: Dataset) -> torch.Tensor:
    try:
        batch_size = batch.ptr.size(dim=0) - 1
    except:
        batch_size = 1
    choices = _masked_argmax(
                pred.view(batch_size, -1),
                batch.neighbors.view(batch_size, -1),
                dim=1
            )
    
    return torch.where(choices < batch.n, choices, -1)


def _greedy_choices(batch: Dataset) -> torch.Tensor:
    try:
        batch_size = batch.ptr.size(dim=0) - 1
    except:
        batch_size = 1
    nodes_per_graph = batch.x.size(dim=0) // batch_size
    all_nodes = torch.arange(0, nodes_per_graph)
    choices = torch.full(fill_value=-1, size=(1, batch_size))

    for i in range(batch_size):
        t = batch.graph_features[2 * i, 0].item()
        n = batch.n[i].item()
        arr_node = t + n
        neighbor_nodes = all_nodes[
            batch.neighbors[i * nodes_per_graph : (i + 1) * nodes_per_graph]
        ]
        
        adj_values = batch.edge_attr[
            (batch.edge_index[0, :] == arr_node) & 
            torch.isin(batch.edge_index[1, :], neighbor_nodes)
        ]
        
        if adj_values.size(dim=0) > 0:
            choices[:, i] = neighbor_nodes[torch.argmax(adj_values, dim=0)]
        
    return choices


class NodeSelector(ABC):
    def __init__(self):
        return

    @abstractmethod
    def select_nodes(self, batches):
        pass


class TorchNodeSelector(NodeSelector):
    def __init__(self, torch_model: object):
        self.model = torch_model
        self.device = next(torch_model.parameters()).device
        super().__init__()

    def select_nodes(self, batches):
        with torch.no_grad():
            choices = []
            for batch in batches:
                batch.to(self.device)
                pred = self.model(batch)
                choices.append(_vtg_greedy_choices(pred, batch))
            return torch.cat(choices)


class GreedyNodeSelector(NodeSelector):
    def __init__(self):
        super().__init__()

    def select_nodes(self, batches):
        choices = []
        for batch in batches:
            batch.to('cpu')
            choices.append(_greedy_choices(batch))
        return torch.cat(choices)
    