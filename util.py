from itertools import chain, combinations
import numpy as np
import pandas as pd
from params import _Array
from torch_geometric.data import InMemoryDataset
from numpy.random import Generator
from copy import copy
import torch
import math
import matplotlib.pyplot as plt
import scipy.stats as st 
import pickle


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


def _masked_argmax(tensor, mask, dim):
    masked = torch.mul(tensor, mask)
    neg_inf = torch.zeros_like(tensor)
    neg_inf[~mask] = -math.inf 
    return (masked + neg_inf).argmax(dim=dim)


def _vtg_greedy_choices(pred: torch.Tensor, batch: Dataset) -> torch.Tensor:
    try:
        batch_size = batch.ptr.size(dim=0) - 1
    except:
        batch_size = 1
    choices = _masked_argmax(
                pred.view(batch_size, -1),
                batch.neighbors.view(batch_size, -1),
                dim=1
            )
    
    return torch.where(choices < batch.n, choices, -1)

def _vtg_predictions(pred: torch.Tensor, batch: Dataset) -> torch.Tensor:
    # Returns the model's predictions, masked to only keep the neighbor predictions
    try:
        batch_size = batch.ptr.size(dim=0) - 1
    except:
        batch_size = 1

    return torch.mul(
        pred.view(batch_size, -1),
        batch.neighbors.view(batch_size, -1)
    )


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

def _load_osmnx(location: str):
    with open(f"data/OSMNX_{location}_travel_times.pickle", "rb") as handle:
        location_info = pickle.load(handle)
    return location_info

def _load_gmission():
    edge_df = pd.read_csv('./data/g_mission/edges.txt', header=None)
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


def _plot_approx_ratios(ratios, data, naming_function = lambda graph_type: graph_type, x_axis_name = "# online / # offline", confidence = 0.99):

    for graph_type, comp_ratios in data.items():
        aggregated_ratios = {}

        for trial_ratios in comp_ratios:
            for model, ratio_values in trial_ratios.items():
                current_ratios = aggregated_ratios.get(model, [])
                
                # Compute the confidence interval for the competitive ratios
                ci_lowerbound, ci_upperbound = st.norm.interval(alpha=confidence, 
                                loc=np.mean(ratio_values), 
                                scale=st.sem(ratio_values)) 
                current_ratios.append((np.array(ratio_values).mean(), ci_lowerbound, ci_upperbound))
                aggregated_ratios[model] = current_ratios

        fig = plt.figure(figsize=(8,6))
        for model, model_ratios in aggregated_ratios.items():
            competitive_ratios = [val[0] for val in model_ratios]
            ci_lowerbounds = [val[1] for val in model_ratios]
            ci_upperbounds = [val[2] for val in model_ratios]
            plt.plot(ratios, competitive_ratios, label=model)
            plt.fill_between(ratios, ci_lowerbounds, ci_upperbounds, alpha = 0.2)

        title = f"{naming_function(graph_type)}"
        plt.title(title, fontsize = 18)
        plt.xlabel(x_axis_name, fontsize = 15)
        plt.ylabel('Average Competitive Ratio', fontsize = 15)
        plt.legend()
        plt.savefig(f"data/{title.replace(' ', '_')}.png")
        plt.show()

def _box_plots(data, naming_function = lambda graph_type: graph_type, colors = None):

    for graph_type, comp_ratios in data.items():
        aggregated_ratios = {}
        labels = comp_ratios.keys()
        fig, ax = plt.subplots(figsize=(6, 6))
        all_data = np.stack(comp_ratios.values()).T
        print(all_data.shape)

        # source: https://matplotlib.org/stable/gallery/statistics/boxplot_color.html#sphx-glr-gallery-statistics-boxplot-color-py
        # rectangular box plot
        bplot = ax.boxplot(all_data,
                            vert=True,  # vertical box alignment
                            patch_artist=True,  # fill with color
                            showfliers=False,  # fill with color
                            )
        for median in bplot['medians']:
            median.set_color('black')

        # fill with colors
        for patch, color in zip(bplot['boxes'], colors):
            patch.set_facecolor(color)

        # adding horizontal grid lines
        ax.yaxis.grid(True)

        title = f"{naming_function(graph_type)}"
        ax.set_title(title, fontsize = 18)
        ax.set_ylabel('Competitive Ratio', fontsize = 15)
        ax.set_xticklabels(labels=labels, fontsize=15)
        plt.legend()
        plt.savefig(f"data/{title.replace(' ', '_')}.png")
        plt.show()