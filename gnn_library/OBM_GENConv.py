import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GENConv
from util import _extract_batch


#TODO: fix graph classification
class OBM_GENConv(torch.nn.Module):
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
        super(OBM_GENConv, self).__init__()

        input_dim = args.node_feature_dim
        output_dim = args.output_dim
        hidden_dim = args.hidden_dim
        edge_feature_dim = args.edge_feature_dim
        self.graph_feature_dim = args.graph_feature_dim
        self.dropout = args.dropout
        self.num_layers = args.num_layers
        self.classify = (args.head in ['classification', 'meta'])

        conv_modules = [GENConv(
            in_channels=input_dim,
            out_channels=hidden_dim,
            aggr=args.aggr,
            edge_dim=edge_feature_dim)]
        conv_modules.extend(
            [
                GENConv(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim,
                    aggr=args.aggr,
                    edge_dim=edge_feature_dim
                )
                for _ in range(self.num_layers - 1)
            ]
        )

        self.convs = nn.ModuleList(conv_modules)
        self.regression_head = nn.Linear(
            hidden_dim + self.graph_feature_dim, output_dim)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        self.regression_head.reset_parameters()
        self.pool.reset_parameters()

    def forward(self, batch):
        x, edge_index, edge_attr, batch, num_graphs, graph_features = \
            _extract_batch(batch)
        
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index, edge_attr)
            x = F.relu(x)
            x = F.dropout(x, self.dropout, self.training)

        if self.graph_feature_dim > 0:
            num_nodes = x.size(dim=0) // num_graphs
            graph_features = torch.cat(
                graph_features.view(
                    num_graphs,
                    self.graph_feature_dim,
                    num_nodes
                ).unbind(dim=0), 
                dim=1
            )
            x = torch.hstack((x, graph_features.T))

        if self.classify:
            x = x.view(num_graphs, num_nodes, -1)[:, -1, :].flatten()
        else:
            x = self.regression_head(x)

        return x
