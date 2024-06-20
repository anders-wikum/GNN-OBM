import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GENConv, global_max_pool
from util import _extract_batch, _vtg_greedy_choices
from torch.nn import LayerNorm, Linear, ReLU
from torch_geometric.nn import DeepGCNLayer

class DeeperGCN(torch.nn.Module):
    def __init__(self, args):
        super().__init__()

        input_dim = args.node_feature_dim
        output_dim = args.output_dim
        hidden_dim = args.hidden_dim
        edge_feature_dim = args.edge_feature_dim
        self.graph_feature_dim = args.graph_feature_dim
        self.dropout = args.dropout
        self.num_layers = args.num_layers
        self.pool = args.head == 'meta'
        self.device = args.device
        self.head = args.head

        self.node_encoder = Linear(input_dim, hidden_dim)
        self.edge_encoder = Linear(edge_feature_dim, hidden_dim)

        self.layers = torch.nn.ModuleList()
        for i in range(1, self.num_layers + 1):
            conv = GENConv(hidden_dim, hidden_dim, aggr='softmax',
                           t=1.0, learn_t=True, num_layers=2, norm='layer')
            norm = LayerNorm(hidden_dim, elementwise_affine=True)
            act = ReLU(inplace=True)

            layer = DeepGCNLayer(conv, norm, act, block='res+', dropout=self.dropout,
                                 ckpt_grad=i % 3)
            self.layers.append(layer)

        self.head_num_layers = args.head_num_layers if 'head_num_layers' in args.__dict__ else 1
        regression_modules = [
            nn.Linear(
                in_features=hidden_dim + self.graph_feature_dim,
                out_features=hidden_dim
            )
        ]
        if self.head_num_layers > 1:
            regression_modules.extend([
                nn.Linear(
                    in_features=hidden_dim,
                    out_features=hidden_dim
                )
                for _ in range(self.head_num_layers - 1)
            ])

        regression_modules.append(
            nn.Linear(
                in_features=hidden_dim,
                out_features=output_dim
            )
        )

        self.regression_head = nn.ModuleList(regression_modules)

    
    def reset_parameters(self):
        if self.classify:
            self.classification_head.reset_parameters()
        for layer in self.regression_head:
            layer.reset_parameters()

    def forward(self, batch):
        x, edge_index, edge_attr, batch_ids, num_graphs, graph_features = \
            _extract_batch(batch)
        
        if self.head == 'meta':
            x = torch.hstack([x, batch.base_model_preds])

        x = self.node_encoder(x)
        edge_attr = self.edge_encoder(edge_attr)

        x = self.layers[0].conv(x, edge_index, edge_attr)

        for layer in self.layers[1:]:
            x = layer(x, edge_index, edge_attr)

        x = self.layers[0].act(self.layers[0].norm(x))
        x = F.dropout(x, p=0.1, training=self.training)


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
            
        x = torch.hstack((x, graph_features.T))
        
     
        if self.pool:
            x = global_max_pool(x, batch_ids)
        
        for i in range(self.head_num_layers):
            x = self.regression_head[i](x)
            x = F.relu(x)
            x = F.dropout(x, self.dropout, self.training)
        x = self.regression_head[-1](x)
        
        if self.pool:
            x = F.softmax(x, dim=1)

        return x

    def batch_select_match_nodes(self, batches):
        with torch.no_grad():
            choices = []
            for batch in batches:
                batch.to(self.device)
                pred = self(batch)
                choices.append(_vtg_greedy_choices(pred, batch))
            return torch.cat(choices)
