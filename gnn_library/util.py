import torch
import copy

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch_geometric.loader import DataLoader
from tqdm import trange
from .params import NETWORKS


class MaskedMSELoss(nn.Module):

    def __init__(self):
        super(MaskedMSELoss, self).__init__()

    def forward(self, pred, value_to_go, neighbor_mask):
        """
        Computes MSE over neighbors of the arriving node.
        Args:
            pred: predicted node embeddings
            value_to_go: array of underlying value to gos
            neighbor_mask: mask for neighbors of arriving node

        Returns:
            Masked mean square error.
        """
        preds = pred[neighbor_mask].squeeze(dim=1)
        match_penalty = 0  # (preds[-1] - value_to_go[-1]) ** 2
        return F.mse_loss(preds, value_to_go) + match_penalty


class BinaryCrossEntropyLoss(nn.Module):

    def __init__(self):
        super(BinaryCrossEntropyLoss, self).__init__()

    def forward(self, pred, label, nieghbors):
        return F.binary_cross_entropy(pred, label)


def build_optimizer(args, params):
    """
    Builds an optimizer according to the given parameters.
    """

    weight_decay = args.weight_decay
    filter_fn = filter(lambda p: p.requires_grad, params)
    if args.opt == 'adam':
        optimizer = optim.Adam(filter_fn, lr=args.lr,
                               weight_decay=weight_decay)
    elif args.opt == 'sgd':
        optimizer = optim.SGD(filter_fn, lr=args.lr,
                              momentum=0.95, weight_decay=weight_decay)
    elif args.opt == 'rmsprop':
        optimizer = optim.RMSprop(
            filter_fn, lr=args.lr, weight_decay=weight_decay)
    elif args.opt == 'adagrad':
        optimizer = optim.Adagrad(
            filter_fn, lr=args.lr, weight_decay=weight_decay)
    if args.opt_scheduler == 'none':
        return None, optimizer
    elif args.opt_scheduler == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=args.opt_decay_step, gamma=args.opt_decay_rate)
    elif args.opt_scheduler == 'cos':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.opt_restart)
    return scheduler, optimizer


class objectview(object):
    def __init__(self, d):
        self.__dict__ = d


def train(train_loader: DataLoader, test_loader: DataLoader, args: dict):
    """
    Trains a GNN model, periodically testing it and accumulating loss values
    Args:
        args: dictionary object containing training parameters
    """

    # Input dimension is 1 (we only have demand information for every node)
    # Edge feature dimension is 2 (capacity and cost per edge)
    # Output dimension is 1 since we predict scalar potential values for each vertex
    model = NETWORKS[args.processor](
        args.node_feature_dim,
        1,
        args.edge_feature_dim,
        args
    )

    if args.head == 'regression':
        loss_fn = MaskedMSELoss()
    elif args.head == 'classification':
        loss_fn = BinaryCrossEntropyLoss()
    else:
        raise NotImplemented

    device = args.device

    _, opt = build_optimizer(args, model.parameters())
    model.to(device)

    # accumulate model performance for plotting
    train_losses = []
    test_losses = []
    best_loss = None
    best_model = None

    for epoch in trange(args.epochs, desc="Training", unit="Epochs"):
        total_loss = 0
        model.train()

        for batch in train_loader:
            batch.to(device)
            opt.zero_grad()
            pred = model(batch.x, batch.edge_index,
                         batch.edge_attr, batch.batch, batch.graph_features)
            loss = loss_fn(pred, batch.hint, batch.neighbors)
            loss.backward()
            opt.step()

            total_loss += loss.item() * batch.num_graphs
        total_loss /= len(train_loader.dataset)
        print(total_loss)
        train_losses.append(total_loss)

        if epoch % 2 == 0:
            test_loss = test(test_loader, model, loss_fn, device)
            print(f'TEST LOSS: {test_loss}')
            test_losses.append(test_loss)
            if best_loss is None or test_loss < best_loss:
                best_loss = test_loss
                best_model = copy.deepcopy(model)
        else:
            test_losses.append(test_losses[-1])

    return train_losses, test_losses, best_model, best_loss


def test(loader, test_model, loss_fn, device):
    test_model.eval()
    test_model.to(device)
    total_loss = 0

    for batch in loader:
        batch.to(device)
        with torch.no_grad():
            pred = test_model(batch.x, batch.edge_index,
                              batch.edge_attr, batch.batch, batch.graph_features)
            loss = loss_fn(pred, batch.hint, batch.neighbors)
            total_loss += loss * batch.num_graphs

    total_loss /= len(loader.dataset)

    return total_loss
