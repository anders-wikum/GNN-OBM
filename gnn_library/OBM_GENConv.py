import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GENConv, global_max_pool


class OBM_GENConv(torch.nn.Module):
    """
    GNN to predict node-level embeddings. Then applies a post-message passing layer to transform into the output
    dimension.
    Part of the code definition is inspired by Colab 2:
    https://colab.research.google.com/drive/1xHmpjVO-Z74NK-dH3qoUBTf-tKUPfOKW?usp=sharing

    The main model used for convolutions is NNConv from the "Dynamic Edge-Conditioned Filters in Convolutional Neural
    Networks on Graphs" <https://arxiv.org/abs/1704.02901> paper
    """

    def __init__(self, input_dim, output_dim, edge_feature_dim, args):
        """
        Initializing the GNN
        Args:
            input_dim: dimension of node features
            output_dim: output dimension required
            edge_feature_dim: dimension of the edge features
            args: object containing the rest of the GNN description, including the number of layers, dropout, ...
        """
        super(OBM_GENConv, self).__init__()

        hidden_dim = args.hidden_dim
        self.batch_size = args.batch_size
        self.graph_feature_dim = args.graph_feature_dim
        self.dropout = args.dropout
        self.num_layers = args.num_layers
        self.classify = (args.head == 'classification')

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

        if self.classify:
            self.pool = global_max_pool

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        self.regression_head.reset_parameters()

    def forward(self, x, edge_index, edge_attr, num_graphs, graph_features):
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
            x = self.regression_head(torch.hstack((x, graph_features.T)))

        else:
            x = self.regression_head(x)

        if self.classify:
            x = F.sigmoid(x.view(num_graphs, -1)[:, -1])
        return x
