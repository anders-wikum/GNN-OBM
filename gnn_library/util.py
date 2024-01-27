import torch
import pickle
import copy

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch_geometric.loader import DataLoader
from tqdm import trange
from .params import NETWORKS, REQ_ARGS, MODEL_SAVE_FOLDER
from util import objectview
from typing import Optional

import optuna


class MaskedMSELoss(nn.Module):

    def __init__(self):
        super(MaskedMSELoss, self).__init__()

    def forward(self, pred, batch):
        """
        Computes MSE over neighbors of the arriving node.
        Args:
            pred: predicted node embeddings
            value_to_go: array of underlying values to go
            neighbor_mask: mask for neighbors of arriving node

        Returns:
            Masked mean square error.
        """
        value_to_go = batch.hint
        neighbor_mask = batch.neighbors
        preds = pred[neighbor_mask].squeeze(dim=1)
        return F.mse_loss(preds, value_to_go)


class pygCrossEntropyLoss(nn.Module):

    def __init__(self):
        super(pygCrossEntropyLoss, self).__init__()

    def forward(self, pred, batch):
        criterion = nn.BCELoss()
        sig = nn.Sigmoid()

        hints = batch.hint

        # Only consider neighbors of the current online node
        neighbor_mask = batch.neighbors
        pred_per_graph = pred[neighbor_mask].squeeze(dim=1).T
        batch_index = batch.batch[neighbor_mask]

        # print(f"pred: {pred.shape}")
        # print(f"batch: {batch.batch.shape}")
        # print(batch.batch[neighbor_mask].shape)

        # print(f"pred per graph before where: {pred_per_graph.shape}")
        # print(f"hints per graph before where: {hints.shape}")

        # Get the predicted node per graph
        mask = torch.arange(len(torch.unique(batch.batch))).unsqueeze(1).to(pred_per_graph.device) == batch_index.unsqueeze(0)
        pred_per_graph = torch.where(mask, pred_per_graph, float('-inf'))

        # print(f"pred per graph before sig: {pred_per_graph.shape}")

        # Create an array that is 1 only 
        maximizers = torch.argmax(torch.where(mask, hints.unsqueeze(0), float('-inf')), dim=1)
        hints_per_graph = torch.zeros_like(pred_per_graph)
        hints_per_graph[torch.arange(maximizers.shape[0]), maximizers] = 1
        
        # Pass predictions through a sigmoid
        # print(f"mask shape {mask.shape}")
        # print(f"hints per graph: {hints_per_graph.shape}")
        pred_per_graph = sig(pred_per_graph)

        # print(pred_per_graph.shape)
        # print(batch.hint)

        #print(torch.argmax(batch.hint.view(-1, C), dim=1), pred)
        # first_elems = torch.nonzero(mask)
        # print(mask)
        # print(first_elems)

        return criterion(
            pred_per_graph,
            hints_per_graph
        )
    

class torchCrossEntropyLoss(nn.Module):

    def __init__(self):
        super(torchCrossEntropyLoss, self).__init__()

    def forward(self, pred, batch):
        (_, y) = batch
        return F.binary_cross_entropy_with_logits(pred, y)


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


def _get_loss(args):
    if args.head == 'regression':
        return MaskedMSELoss()
    elif args.head == 'classification':
        return pygCrossEntropyLoss()
    elif args.head == 'meta':
        if args.processor == 'NN':
            return torchCrossEntropyLoss()
        else:
            return pygCrossEntropyLoss()
    else:
        raise NotImplemented

def _get_acc(args):
    if args.head == 'regression':
        return _regression_accuracy
    elif args.head == 'classification':
        return _classification_accuracy
    else:
        raise NotImplemented
    

def _get_model(args: dict):
    model = NETWORKS[args.processor](args)
    return model


def train(train_loader, test_loader, args, trial = None):
    args = objectview(args)
    model = _get_model(args)
    model.to(args.device)
    loss_fn = _get_loss(args)
    acc_fn = _get_acc(args)
    _, opt = build_optimizer(args, model.parameters())

    return _train(
        model=model,
        loss_fn=loss_fn,
        acc_fn=acc_fn,
        train_loader=train_loader,
        test_loader=test_loader,
        epochs=args.epochs,
        opt=opt,
        device=args.device,
        trial=trial
    )

def _regression_accuracy(pred, batch):

    # Only consider neighbors of the current online node
    hints = batch.hint
    neighbor_mask = batch.neighbors
    preds = pred[neighbor_mask].squeeze(dim=1)
    batch_index = batch.batch[neighbor_mask]

    # Get the predicted node per graph
    mask = torch.arange(len(torch.unique(batch.batch))).unsqueeze(1).to(preds.device) == batch_index.unsqueeze(0)
    pred_per_graph = torch.argmax(torch.where(mask, preds.unsqueeze(0), float('-inf')), dim=1)
    hints_per_graph = torch.argmax(torch.where(mask, hints.unsqueeze(0), float('-inf')), dim=1)

    return torch.sum(pred_per_graph == hints_per_graph) / pred_per_graph.shape[0]

def _classification_accuracy(pred, batch):

    # Only consider neighbors of the current online node
    hints = batch.hint
    neighbor_mask = batch.neighbors
    preds = pred[neighbor_mask].squeeze(dim=1)
    batch_index = batch.batch[neighbor_mask]

    # Get the predicted node per graph
    mask = torch.arange(len(torch.unique(batch.batch))).unsqueeze(1).to(preds.device) == batch_index.unsqueeze(0)
    pred_per_graph = torch.argmax(torch.where(mask, preds.unsqueeze(0), float('-inf')), dim=1)
    hints = torch.argmax(torch.where(mask, hints.unsqueeze(0), float('-inf')), dim=1)
    # hints = batch.hint.type(torch.LongTensor).to(pred_per_graph.device)

    return torch.sum(pred_per_graph == hints) / pred_per_graph.shape[0]

def _train(
        model: object,
        loss_fn: callable,
        acc_fn: callable,
        train_loader: DataLoader,
        test_loader: DataLoader,
        epochs: int,
        opt: optim.Optimizer,
        device: str,
        trial=None
        ):
    """
    Trains a GNN model, periodically testing it and accumulating loss values
    Args:
        args: dictionary object containing training parameters
    """

    # accumulate model performance for plotting
    train_losses = []
    test_losses = []
    test_accuracies = []
    best_loss = None
    best_model = None

    for epoch in trange(epochs, desc="Training", unit="Epochs"):
        total_loss = 0
        model.train()

        for batch in train_loader:
            if type(batch) is list:
                batch = (batch[0].to(device), batch[1].to(device))
                scale = batch[0].size(dim=0)
            else:
                batch.to(device)
                scale = batch.num_graphs
            opt.zero_grad()

            pred = model(batch)
            loss = loss_fn(pred, batch)

            loss.backward()
            opt.step()

            total_loss += loss.item() * scale
        total_loss /= len(train_loader.dataset)
        if trial is None:
            print(f"TRAINING LOSS: {total_loss}")
        train_losses.append(total_loss)

        if epoch % 4 == 0:
            test_loss, test_accuracy = _test(test_loader, model, loss_fn, acc_fn, device)
            if trial is None:
                print(f'TEST LOSS: {test_loss}')
                print(f'TEST ACCURACY: {test_accuracy}')
            test_losses.append(test_loss)
            test_accuracies.append(test_accuracy)
            if best_loss is None or test_loss < best_loss:
                best_loss = test_loss
                best_model = copy.deepcopy(model)

            # Report the test to the tuner
            if trial is not None:
                trial.report(test_accuracy, epoch)

            # Prune based on the intermediate value
            if trial is not None and trial.should_prune():
                raise optuna.exceptions.TrialPruned()
        else:
            test_losses.append(test_losses[-1])
            test_accuracies.append(test_accuracies[-1])

    return train_losses, test_losses, test_accuracies, best_model, best_loss


def _test(loader, test_model, loss_fn, acc_fn, device):
    test_model.eval()
    test_model.to(device)
    total_loss = 0
    total_accuracy = 0

    for batch in loader:
        if type(batch) is list:
            batch = (batch[0].to(device), batch[1].to(device))
            scale = batch[0].size(dim=0)
        else:
            batch.to(device)
            scale = batch.num_graphs
        with torch.no_grad():
            pred = test_model(batch)
            loss = loss_fn(pred, batch)
            total_loss += loss * scale
            total_accuracy += acc_fn(pred, batch) * scale

    total_loss /= len(loader.dataset)
    total_accuracy /= len(loader.dataset)

    return total_loss, total_accuracy

def save(model: object, args: dict, name: str) -> None:
    path = MODEL_SAVE_FOLDER + name
    #filtered_args = {key: args[key] for key in REQ_ARGS[args['head']]}
    torch.save(model.state_dict(), path)
    pickle.dump(args, open(path + '_args.pickle', 'wb'))

def load(name: str, device: str) -> object:
    path = MODEL_SAVE_FOLDER + name
    args = pickle.load(open(path + '_args.pickle', 'rb'))
    args['device'] = device
    args = objectview(args)
    model = _get_model(args)
    model.load_state_dict(torch.load(path, map_location='cpu'))
    model.to(device)
    model.eval()
    return model, args
