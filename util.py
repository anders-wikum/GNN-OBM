from itertools import chain, combinations
import numpy as np
import pandas as pd
from params import _Array
from torch_geometric.data import InMemoryDataset
from numpy.random import Generator
from copy import copy
import torch
from typing import List
from params import _Instance


class Dataset(InMemoryDataset):
    def __init__(self, data_list):
        super().__init__(None)
        self.data, self.slices = self.collate(data_list)

class NumpyDataset(Dataset):
    def __init__(self, data_list):
        X = []
        y = []
        for data in data_list:
            X.append(data.X)
            y.append(data.y)

        self.x = torch.FloatTensor(np.vstack(X))
        self.y = torch.FloatTensor(np.vstack(y))
        
    def __getitem__(self, index):
        x = self.x[index]
        y = self.y[index]       
        return (x, y)
    
    def __len__(self):
        return len(self.x)
    

class objectview(object):
    def __init__(self, d):
        self.__dict__ = d


def fill_list(value: object, size: int):
    return [copy(value) for _ in range(size)]


def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    powerset = chain.from_iterable(combinations(s, r) for r in range(len(s)+1))
    return list(map(set, powerset))


def diff(S: frozenset, u: int) -> frozenset:
    return S.difference(set([u]))


def _random_subset(seq, m, rng: Generator):
    """Return m unique elements from seq.

    This differs from random.sample which can return repeated
    elements if seq holds repeated elements.
    """
    targets = set()
    while len(targets) < m:
        x = rng.choice(seq)
        targets.add(x)

    return targets


def _symmetrize(adj):
    n, m = adj.shape
    A = np.zeros((n + m, n + m))
    A[:n, n:] = adj
    A[n:, :n] = adj.T
    return A


def _neighbors(A: _Array, S: set, t: int) -> _Array:
    return [u for u in S if A[t, u] > 0]


def _load_gmission():
    edge_df = pd.read_csv('./data/edges.txt', header=None)
    edge_df[['worker_type', 'task_type']
            ] = edge_df[0].str.split(';', expand=True)
    edge_df = edge_df \
        .drop(columns=[0]) \
        .rename(columns={1: 'weight'}) \
        .assign(worker_type=lambda x: x['worker_type'].apply(float).apply(int)) \
        .assign(task_type=lambda x: x['task_type'].apply(float).apply(int)) \
        .assign(weight=lambda x: x['weight'].apply(float))

    # Normalize edge weights
    edge_df['weight'] = (edge_df['weight'] - edge_df['weight'].min()) \
        / (edge_df['weight'].max() - edge_df['weight'].min())

    return edge_df


def _extract_batch(batch):
    try:
        num_graphs = batch.ptr.size(dim=0) - 1
    except:
        num_graphs = 1
    return (
        batch.x,
        batch.edge_index,
        batch.edge_attr,
        batch.batch, 
        num_graphs,
        batch.graph_features
    )


def _flip_coins(p: _Array, rng: Generator) -> _Array:
    return np.vectorize(lambda x: rng.binomial(1, x))(p)
