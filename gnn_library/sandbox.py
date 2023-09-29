import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GENConv, global_mean_pool, global_max_pool
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

        return x.view(num_graphs, -1).softmax(dim=1)
    
    def batch_select_match_nodes(self, batches):
        with torch.no_grad():
            choices = []
            for batch in batches:
                batch.to(self.device)
                pred = self(batch)
                choices.append(_vtg_greedy_choices(pred, batch))
            return torch.cat(choices)


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
        self.classify = args.head == 'classification'
        self.pool = args.head == 'meta'
        self.device = args.device

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
        x, edge_index, edge_attr, batch, num_graphs, graph_features = \
            _extract_batch(batch)
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
        
        if self.classify:
            x = x.view(num_graphs, num_nodes, -1)[:, -1, :].flatten()
        elif self.pool:
            # Pooling before regression
            x = global_max_pool(x, batch)
        
        if not self.classify: 
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

from torch_geometric.nn import GATConv

class GAT(torch.nn.Module):
    def __init__(self, args):
        super().__init__()

        input_dim = args.node_feature_dim
        output_dim = args.output_dim
        hidden_dim = args.hidden_dim
        edge_feature_dim = args.edge_feature_dim
        self.graph_feature_dim = args.graph_feature_dim
        self.dropout = args.dropout
        self.num_layers = args.num_layers
        self.classify = args.head == 'classification'
        self.pool = args.head == 'meta'
        self.device = args.device

        self.node_encoder = Linear(input_dim, hidden_dim)
        self.edge_encoder = Linear(edge_feature_dim, hidden_dim)

        self.layers = torch.nn.ModuleList()
        for i in range(1, self.num_layers + 1):
            conv = GATConv(hidden_dim, hidden_dim, dropout = self.dropout)
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
        x, edge_index, edge_attr, batch, num_graphs, graph_features = \
            _extract_batch(batch)
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
        
        if self.classify:
            x = x.view(num_graphs, num_nodes, -1)[:, -1, :].flatten()
        elif self.pool:
            # Pooling before regression
            x = global_max_pool(x, batch)
        
        if not self.classify: 
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


from torch_geometric.nn import GATv2Conv

class GATv2Conv(torch.nn.Module):
    def __init__(self, args):
        super().__init__()

        input_dim = args.node_feature_dim
        output_dim = args.output_dim
        hidden_dim = args.hidden_dim
        edge_feature_dim = args.edge_feature_dim
        self.graph_feature_dim = args.graph_feature_dim
        self.dropout = args.dropout
        self.num_layers = args.num_layers
        self.classify = args.head == 'classification'
        self.pool = args.head == 'meta'
        self.device = args.device

        self.node_encoder = Linear(input_dim, hidden_dim)
        self.edge_encoder = Linear(edge_feature_dim, hidden_dim)

        self.layers = torch.nn.ModuleList()
        for i in range(1, self.num_layers + 1):
            conv = GATv2Conv(hidden_dim, hidden_dim, dropout = self.dropout)
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
        x, edge_index, edge_attr, batch, num_graphs, graph_features = \
            _extract_batch(batch)
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
        
        if self.classify:
            x = x.view(num_graphs, num_nodes, -1)[:, -1, :].flatten()
        elif self.pool:
            # Pooling before regression
            x = global_max_pool(x, batch)
        
        if not self.classify: 
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

from torch_geometric.nn import NNConv

class NNConvSandbox(torch.nn.Module):
    def __init__(self, args):
        super().__init__()

        input_dim = args.node_feature_dim
        output_dim = args.output_dim
        hidden_dim = args.hidden_dim
        edge_feature_dim = args.edge_feature_dim
        self.graph_feature_dim = args.graph_feature_dim
        self.dropout = args.dropout
        self.num_layers = args.num_layers
        self.classify = args.head == 'classification'
        self.pool = args.head == 'meta'
        self.device = args.device

        self.node_encoder = Linear(input_dim, hidden_dim)
        self.edge_encoder = Linear(edge_feature_dim, hidden_dim)

        self.layers = torch.nn.ModuleList()
        for i in range(1, self.num_layers + 1):
            conv = NNConv(hidden_dim, hidden_dim, dropout = self.dropout)
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
        x, edge_index, edge_attr, batch, num_graphs, graph_features = \
            _extract_batch(batch)
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
        
        if self.classify:
            x = x.view(num_graphs, num_nodes, -1)[:, -1, :].flatten()
        elif self.pool:
            # Pooling before regression
            x = global_max_pool(x, batch)
        
        if not self.classify: 
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

from torch_geometric.nn import TransformerConv

class TransformerConvSandbox(torch.nn.Module):
    def __init__(self, args):
        super().__init__()

        input_dim = args.node_feature_dim
        output_dim = args.output_dim
        hidden_dim = args.hidden_dim
        edge_feature_dim = args.edge_feature_dim
        self.graph_feature_dim = args.graph_feature_dim
        self.dropout = args.dropout
        self.num_layers = args.num_layers
        self.classify = args.head == 'classification'
        self.pool = args.head == 'meta'
        self.device = args.device

        self.node_encoder = Linear(input_dim, hidden_dim)
        self.edge_encoder = Linear(edge_feature_dim, hidden_dim)

        self.layers = torch.nn.ModuleList()
        for i in range(1, self.num_layers + 1):
            conv = TransformerConv(hidden_dim, hidden_dim, dropout = self.dropout)
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
        x, edge_index, edge_attr, batch, num_graphs, graph_features = \
            _extract_batch(batch)
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
        
        if self.classify:
            x = x.view(num_graphs, num_nodes, -1)[:, -1, :].flatten()
        elif self.pool:
            # Pooling before regression
            x = global_max_pool(x, batch)
        
        if not self.classify: 
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

from torch_geometric.nn import SuperGATConv

class SuperGATConv(torch.nn.Module):
    def __init__(self, args):
        super().__init__()

        input_dim = args.node_feature_dim
        output_dim = args.output_dim
        hidden_dim = args.hidden_dim
        edge_feature_dim = args.edge_feature_dim
        self.graph_feature_dim = args.graph_feature_dim
        self.dropout = args.dropout
        self.num_layers = args.num_layers
        self.classify = args.head == 'classification'
        self.pool = args.head == 'meta'
        self.device = args.device

        self.node_encoder = Linear(input_dim, hidden_dim)
        self.edge_encoder = Linear(edge_feature_dim, hidden_dim)

        self.layers = torch.nn.ModuleList()
        for i in range(1, self.num_layers + 1):
            conv = SuperGATConv(hidden_dim, hidden_dim, dropout = self.dropout)
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
        x, edge_index, edge_attr, batch, num_graphs, graph_features = \
            _extract_batch(batch)
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
        
        if self.classify:
            x = x.view(num_graphs, num_nodes, -1)[:, -1, :].flatten()
        elif self.pool:
            # Pooling before regression
            x = global_max_pool(x, batch)
        
        if not self.classify: 
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