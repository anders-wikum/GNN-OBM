import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphClassifier(torch.nn.Module):
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
        super(GraphClassifier, self).__init__()

        self.input_dim = args.input_dim
        self.hidden_dim = args.hidden_dim
        self.output_dim = args.output_dim
        self.dropout = args.dropout
        self.num_layers = args.num_layers

        modules = [
            nn.Linear(
                in_features=self.input_dim,
                out_features=self.hidden_dim
            )
        ]
        modules.extend(
            [
                nn.Linear(
                    in_features=self.hidden_dim,
                    out_features=self.hidden_dim
                )
                for _ in range(self.num_layers - 1)
            ]
        )
        modules.extend(
            [
                nn.Linear(
                    in_features=self.hidden_dim,
                    out_features=self.output_dim
                )
            ]
        )
        

        self.mods = nn.ModuleList(modules)


    def reset_parameters(self):
        for layer in self.mods:
            layer.reset_parameters()

    def forward(self, batch):
        (x, _) = batch
        for i in range(self.num_layers):
            x = self.mods[i](x)
            x = F.relu(x)
            x = F.dropout(x, self.dropout, self.training)
        x = self.mods[-1](x)
        return x
