import torch
from util import Dataset

def _greedy_choices(batch: Dataset, t: float) -> torch.Tensor:
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
        arr_node = t + n + i * nodes_per_graph
        neighbor_nodes = all_nodes[
            batch.neighbors[i * nodes_per_graph : (i + 1) * nodes_per_graph]
        ]

        adj_values = batch.edge_attr[
            (batch.edge_index[0, :] == arr_node) & 
            torch.isin(
                batch.edge_index[1, :],
                neighbor_nodes + i * nodes_per_graph
            )
        ]
        
        if adj_values.size(dim=0) > 0 and torch.max(adj_values) >= t:
            choices[:, i] = neighbor_nodes[torch.argmax(adj_values, dim=0)]
    
    return torch.where(choices < batch.n, choices, -1)[0]



class OBM_Greedy(torch.nn.Module):
    """
    Torch module wrapper for greedy online bipartite matching algorithm. Allows
    for easy comparison to neural net-based approaches.
    """

    def __init__(self, t):
        """
        Initializing the GNN
        Args:
            None
        """
        super(OBM_Greedy, self).__init__()
        self.eval()
        self.t = t

    def reset_parameters(self):
        pass

    def forward(self, batch):
        pass

    def batch_select_match_nodes(self, batches):
        choices = []
        for batch in batches:
            batch.to('cpu')
            choices.append(_greedy_choices(batch, self.t))
        return torch.cat(choices)
