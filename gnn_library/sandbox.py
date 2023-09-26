import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GENConv, global_mean_pool
from util import _extract_batch, _vtg_greedy_choices


class OBM_class(torch.nn.Module):
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
        super(OBM_class, self).__init__()

        input_dim = args.node_feature_dim
        output_dim = args.output_dim
        hidden_dim = args.hidden_dim
        edge_feature_dim = args.edge_feature_dim
        self.graph_feature_dim = args.graph_feature_dim
        self.dropout = args.dropout
        self.num_layers = args.num_layers
        self.classify = (args.head in ['classification', 'meta'])
        self.device = args.device

        conv_modules = [
            GENConv(
                in_channels=input_dim,
                out_channels=hidden_dim,
                aggr=args.aggr,
                edge_dim=edge_feature_dim
            )
        ]
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
            hidden_dim + self.graph_feature_dim,
            output_dim
        )

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        self.regression_head.reset_parameters()

    def forward(self, batch):
        x, edge_index, edge_attr, batch, num_graphs, graph_features = \
            _extract_batch(batch)

        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index, edge_attr)
            x = F.relu(x)
            x = F.dropout(x, self.dropout, self.training)

        x = global_mean_pool(x, batch)
        if self.graph_feature_dim > 0:
            num_nodes = x.size(dim=0) // num_graphs
            graph_features = graph_features.view(
                num_graphs,
                self.graph_feature_dim
            )
        
        #print(graph_features.size(), x.size())
        x = torch.hstack((x, graph_features))
        #print(x)
        x = self.regression_head(x)

        
        # if self.classify:
        #     x = x.view(num_graphs, num_nodes, -1)[:, -1, :].flatten()       

        return x.view(num_graphs, -1)#.softmax(dim=1)
    
    def batch_select_match_nodes(self, batches):
        with torch.no_grad():
            choices = []
            for batch in batches:
                batch.to(self.device)
                pred = self(batch)
                choices.append(_vtg_greedy_choices(pred, batch))
            return torch.cat(choices)
