import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import NNConv


class OBM_NNConv(torch.nn.Module):
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
        super(OBM_NNConv, self).__init__()

        hidden_dim = args.hidden_dim
        self.dropout = args.dropout
        aggr = args.aggr
        self.num_layers = args.num_layers
        conv_modules = [NNConv(input_dim, hidden_dim, nn.Linear(
            edge_feature_dim, input_dim * hidden_dim), aggr=aggr)]
        conv_modules.extend(
            [NNConv(hidden_dim, hidden_dim, nn.Linear(edge_feature_dim, hidden_dim * hidden_dim), aggr=aggr) for _ in
             range(self.num_layers - 1)])

        self.convs = nn.ModuleList(conv_modules)
        self.regression_head = nn.Linear(
            hidden_dim + args.graph_feature_dim, output_dim)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        self.regression_head.reset_parameters()

    def forward(self, x, edge_index, edge_attr, graph_features):
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index, edge_attr)
            x = F.relu(x)
            x = F.dropout(x, self.dropout, self.training)
        return self.regression_head(torch.hstack((x, graph_features.T)))
