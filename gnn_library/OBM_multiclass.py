import torch
import torch.nn as nn
import torch.nn.functional as F
from util import _vtg_greedy_choices


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
        self.device = args.device

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
    
    def batch_select_match_nodes(self, batches):
        with torch.no_grad():
            choices = []
            for batch in batches:
                batch.to(self.device)
                pred = self(batch)
                choices.append(_vtg_greedy_choices(pred, batch))
            return torch.cat(choices)
