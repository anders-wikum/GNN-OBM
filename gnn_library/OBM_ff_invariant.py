import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GENConv
from util import _extract_batch, _vtg_greedy_choices, _vtg_predictions


class OBM_ff_invariant(torch.nn.Module):
    """
    GNN to predict node-level embeddings. Then applies a linear layer to 
    transform into the output dimension.
    """

    def __init__(self, args):
        """
        Initializing the GNN
        Args:
            input_dim: dimension of node features
            output_dim: output dimension required (1 for regression/
                classification tasks)
            edge_feature_dim: dimension of the edge features
            args: object containing the rest of the GNN description, including
                the number of layers, dropout, etc.
        """
        super(OBM_ff_invariant, self).__init__()

        input_dim = args.node_feature_dim
        output_dim = args.output_dim
        hidden_dim = args.hidden_dim
        edge_feature_dim = args.edge_feature_dim
        self.graph_feature_dim = args.graph_feature_dim
        self.dropout = args.dropout
        self.num_layers = args.num_layers
        self.classify = (args.head in ['classification', 'meta'])
        self.device = args.device
        self.head = args.head

        self.ff = nn.Sequential(
            nn.Linear(input_dim + self.graph_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def reset_parameters(self):
        self.ff.reset_parameters()

    def forward(self, batch):
        x, edge_index, edge_attr, _, num_graphs, graph_features = \
            _extract_batch(batch)
        print(f"num graphs: {num_graphs}")
        print(f"edge_index shape: {edge_index.shape}")
        print(f"edge_attr shape: {edge_attr.shape}")
        print(f"edge_index shape: {edge_index}")
        print(f"edge_attr shape: {edge_attr}")


        

        if self.graph_feature_dim > 0:
            num_nodes = x.size(dim=0) // num_graphs
            graph_features = graph_features \
                .T \
                .unsqueeze(self.graph_feature_dim) \
                .repeat(1, 1, num_nodes)
           
            graph_features = torch.cat(
                graph_features.view(
                    num_graphs,
                    self.graph_feature_dim,
                    num_nodes
                ).unbind(dim=0), 
                dim=1
            )

            print(f"graph features shape: {graph_features.shape}")
            print(f"graph features: {graph_features}")

        
        x = torch.hstack((x, graph_features.T))

        # x = self.ff(x, edge_index, edge_attr)
        x = self.ff(x)

        if self.classify:
            x = x.view(num_graphs, num_nodes, -1)[:, -1, :]    

        return x
    
    def batch_select_match_nodes(self, batches):
        with torch.no_grad():
            choices = []
            for batch in batches:
                batch.to(self.device)
                pred = self(batch)
                choices.append(_vtg_greedy_choices(pred, batch))
            return torch.cat(choices)