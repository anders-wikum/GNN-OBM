import ast
import math
import matplotlib
import pickle
import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as st 

from itertools import chain, combinations
from matplotlib.ticker import MultipleLocator, AutoMinorLocator, FuncFormatter
from numpy.random import Generator
from torch_geometric.data import InMemoryDataset

from params import _Array


Tensor = torch.Tensor

matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams['text.usetex'] = False


class Dataset(InMemoryDataset):
    def __init__(self, data_list):
        super().__init__(None)
        self.data, self.slices = self.collate(data_list)


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

def representative_gather(input: Tensor, dim: int, index: Tensor) -> Tensor:
    indices = index.unsqueeze(1).unsqueeze(2)
    indices = torch.repeat_interleave(indices, input.size(1), dim=1)
    return input.gather(dim=dim, index=indices).squeeze().T
    return input[torch.arange(input.size(0)), :, index]

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
        batch.t
        #batch.graph_features
    )


def _flip_coins(p: _Array, rng: Generator) -> _Array:
    return rng.binomial(1, p)


def graph_config_to_string(config):
    graph_type = config['graph_type']
    if graph_type == 'ER':
        return f"ER_{config['p']}"
    if graph_type == 'BA':
        return f"BA_{config['ba_param']}"
    if graph_type == 'GEOM':
        return f"GEOM_{config['q']}"
    if graph_type == 'GM':
        return "GM"
    if graph_type == 'OSMNX':
        return f"OSMNX_{config['location']}"

label_map = {
    'meta_gnn': 'MAGNOLIA',
    'learned': 'MAGNOLIA',
    'greedy': 'greedy',
    'greedy_t': 'greedy-t',
    'lp_rounding': 'LP-rounding',
    '(5,3)': '(3$\\times$5)',
    '(10,6)': '(6$\\times$10)',
    '(15,9)': '(9$\\times$15)',
    '(20,12)': '(12$\\times$20)',
    'naor_lp_rounding': 'LP-rounding (Naor)',
    'pollner_lp_rounding': 'LP-rounding (Pollner)',
    'GENConv': 'GENConv',
    'DeeperGCN': 'DeeperGCN',
    'GATv2Conv': 'GATv2Conv',
    'GraphConv': 'GraphConv',
    'GCNConv': 'GCNConv',
    'meta_threshold': 'threshold (meta)',
    'GNN1': 'GNN1',
    'GNN2': 'GNN2',
}

color_map = {
    'meta_gnn': '#ff1f5b',
    'learned': '#ff1f5b',
    'greedy': '#009ade',
    'greedy_t': '#af58ba',
    'lp_rounding': '#00cd6c',
    'naor_lp_rounding': '#F9812A',
    'pollner_lp_rounding': '#009ade',
    '(5,3)': '#00cd6c',
    '(10,6)': '#ff1f5b',
    '(15,9)': '#af58ba',
    '(20,12)': '#009ade',
    'GENConv': '#ff1f5b',
    'DeeperGCN': '#009ade',
    'GATv2Conv': '#af58ba',
    'GraphConv': '#00cd6c',
    'GCNConv': '#ff1f5b',
    'meta_threshold': '#F9812A',
    'GNN1': '#009ade',
    'GNN2': '#af58ba',
}

def title_of_graph_type2(graph_type):
    if 'ER' in graph_type:
        return f"ER, p={graph_type[3:]}"
    if 'GM' in graph_type:
        return "gMission"
    if 'OSMNX' in graph_type:
        return f"Rideshare, {graph_type[6:-17]}"
    if 'BA' in graph_type:
        return f"BA, b={graph_type[3:]}"
    else:
        return f"b-RGG, q={graph_type[5:]}"
    

def title_of_graph_type(graph_type):
    if type(graph_type) == str:
        graph = ast.literal_eval(graph_type)
    else:
        graph = dict(graph_type)
    if graph['graph_type'] == 'ER':
        return f"ER, p={graph['p']}"
    if graph['graph_type'] == 'GM':
        return "gMission"
    if graph['graph_type'] == 'OSMNX':
        return f"Rideshare, {graph['location'].split(',',1)[0]}"
    if graph['graph_type'] == 'BA':
        return f"BA, b={graph['ba_param']}"
    else:
        return f"b-RGG, q={graph['q']}"

model_order = label_map.keys()

def _plot_approx_ratios(ratios, data, naming_function = lambda graph_type: graph_type, x_axis_name = "# online / # offline", confidence = 0.95):
    fontsize = 17
    fontsize2 = 20

    num_subplots = len(data.keys())
    fig, ax = plt.subplots(1, num_subplots, sharex=True, sharey=True, figsize=(12,3))
    fig.add_subplot(111, frameon=False)

    i = 0
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

        for model, model_ratios in aggregated_ratios.items():
            # if model in ['learned', 'greedy', 'greedy_t', 'lp_rounding']:
            competitive_ratios = [val[0] for val in model_ratios]
            ci_lbs = [val[1] for val in model_ratios]
            ci_ubs = [val[2] for val in model_ratios]
            ax[i].plot(ratios, competitive_ratios, label=label_map[model], color=color_map[model])
            ax[i].fill_between(ratios, ci_lbs, ci_ubs, alpha = 0.2, color=color_map[model])

            ax[i].xaxis.set_major_locator(MultipleLocator(50))
            # ax[i].yaxis.set_major_locator(MultipleLocator(0.05))

            ax[i].tick_params(axis='both', which='major', labelsize=13)
            ax[i].tick_params(axis='both', which='minor', labelsize=13)

            ax[i].grid(visible=True, which='both', axis='both')
            ax[i].set_title(title_of_graph_type(graph_type), fontsize=fontsize)
            #ax[i].set_ylim([0.76, 1.01])
            handles, labels = ax[i].get_legend_handles_labels()
        i += 1

    order = [0, 1, 2, 3]
    fig.legend(
        [handles[idx] for idx in order],
        [labels[idx] for idx in order],
        bbox_to_anchor=(1.08, 0.25),
        loc='lower right',
        fontsize=15
    )
    plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
    plt.xlabel(x_axis_name, fontsize = fontsize2 - 2, labelpad=15)
    plt.ylabel('Average competitive ratio', fontsize = fontsize2 - 2, labelpad=20)
    plt.savefig(f"plots/diff_gnn_main.pdf", dpi=300, bbox_inches = "tight")
    plt.show()
 
def _plot_approx_ratios_all(ratios, data, naming_function = lambda graph_type: graph_type, x_axis_name = "# online / # offline", confidence = 0.95):
    k = 4
    fontsize = 20
    fontsize2 = 20
    num_subplots = len(data.keys())
    fig, ax = plt.subplots(k, num_subplots//k, sharex=True, sharey=True, figsize=(12,16))
    fig.add_subplot(111, frameon=False)

    i = 0
    j = 0
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


        for model, model_ratios in aggregated_ratios.items():
            # if model in ['learned', 'greedy', 'greedy_t', 'lp_rounding']:
                competitive_ratios = [val[0] for val in model_ratios]
                ci_lbs = [val[1] for val in model_ratios]
                ci_ubs = [val[2] for val in model_ratios]
                ax[i, j].plot(ratios, competitive_ratios, label=label_map[model], color=color_map[model])
                ax[i, j].tick_params(axis='both', which='major', labelsize=13)
                ax[i, j].tick_params(axis='both', which='minor', labelsize=13)

                # ax[i, j].xaxis.set_major_locator(MultipleLocator(0.2))
                # ax[i, j].yaxis.set_major_locator(MultipleLocator(0.1))
                ax[i, j].xaxis.set_major_locator(MultipleLocator(50))
                # ax[i, j].yaxis.set_major_locator(MultipleLocator(0.02))

                ax[i, j].fill_between(ratios, ci_lbs, ci_ubs, alpha = 0.2, color=color_map[model])
                ax[i, j].grid(visible=True, which='both', axis='both')
                ax[i, j].set_title(title_of_graph_type(graph_type), fontsize=fontsize - 2)
   

        handles, labels = ax[i, j].get_legend_handles_labels()

        j += 1
        j %= num_subplots // k
        if j == 0:
            i += 1

    order = [0, 1, 2, 3]
    fig.legend(
        [handles[idx] for idx in order],
        [labels[idx] for idx in order],
        bbox_to_anchor=(1.15, 0.440),
        loc='lower right',
        fontsize=fontsize
    )
    plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False, labelsize=fontsize)
    plt.xlabel(x_axis_name, fontsize = fontsize2, labelpad=15)
    plt.ylabel('Average competitive ratio', fontsize = fontsize2, labelpad=15)
    space = 0.15
    plt.subplots_adjust(wspace=space, hspace=space)
    plt.savefig(f"plots/diff_training_size_generalization_all.pdf", dpi=300, bbox_inches = "tight")
    plt.show()

def _plot_approx_ratios_single(ratios, data, naming_function = lambda graph_type: graph_type, x_axis_name = "# online / # offline", confidence = 0.95):
    num_subplots = len(data.keys())
    fig, ax = plt.subplots(1, num_subplots, sharex=True, sharey=True, figsize=(12,12))
    fig.add_subplot(111, frameon=False)

    i = 0
    for graph_type, comp_ratios in data.items():
        aggregated_ratios = {}

        for trial_ratios in comp_ratios:
            for model, ratio_values in trial_ratios.items():
                current_ratios = aggregated_ratios.get(model, [])
                
                ci_lowerbound, ci_upperbound = st.norm.interval(alpha=confidence, 
                                loc=np.mean(ratio_values), 
                                scale=st.sem(ratio_values)) 
                current_ratios.append((np.array(ratio_values).mean(), ci_lowerbound, ci_upperbound))
                aggregated_ratios[model] = current_ratios


        for model, model_ratios in aggregated_ratios.items():
            competitive_ratios = [val[0] for val in model_ratios]
            ci_lbs = [val[1] for val in model_ratios]
            ci_ubs = [val[2] for val in model_ratios]
            ax.plot(ratios, competitive_ratios, label=label_map[model], color=color_map[model])
            ax.fill_between(ratios, ci_lbs, ci_ubs, alpha = 0.2, color=color_map[model])
            ax.grid(visible=True, which='both', axis='both')
            ax.set_title(title_of_graph_type(graph_type))
            #ax[i].set_ylim([0.76, 1.01])
            handles, labels = ax.get_legend_handles_labels()
        i += 1

    order = [0, 1]
    print(labels)

    plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
    plt.xlabel(x_axis_name, fontsize = 15, labelpad=15)
    plt.ylabel('Average competitive ratio', fontsize = 15, labelpad=15)
    plt.savefig(f"plots/noise_robustness_all.pdf", dpi=300, bbox_inches = "tight")
    plt.show()

def _extract_models(data):
    for comp_ratios in data.values():
        return [model for model in model_order if model in comp_ratios] 


def _box_plots(data, naming_function = lambda graph_type: graph_type):
    num_subplots = len(data.keys())
    fig, ax = plt.subplots(1, num_subplots, sharex=True, sharey=True, figsize=(12,3))
    #fig, ax = plt.subplots(1, num_subplots, sharex=True, sharey=True, figsize=(6,3))
    fig.add_subplot(111, frameon=False)

    models = _extract_models(data)
    i = 0
    for graph_type, comp_ratios in data.items():
        all_data = np.stack([comp_ratios[model] for model in models]).T

        # source: https://matplotlib.org/stable/gallery/statistics/boxplot_color.html#sphx-glr-gallery-statistics-boxplot-color-py
        # rectangular box plot
        bplot = ax[i].boxplot(all_data,
                            vert=True,  # vertical box alignment
                            patch_artist=True,  # fill with color
                            showfliers=False,  # fill with color
                            # whis=0
                            )
        for median in bplot['medians']:
            median.set_color('black')

        # fill with colors
        colors = [color_map[model] for model in models]
        for patch, color in zip(bplot['boxes'], colors):
            patch.set_facecolor(color)

        # adding horizontal grid lines
        ax[i].grid(visible=True, which='both', axis='both')
        ax[i].set_title(title_of_graph_type(graph_type), fontsize=17)
        ax[i].set_xticklabels([])
        # ax[i].set_ylim(bottom = 0.72, top = 1.01)
        ax[i].tick_params(labelcolor='none', axis = 'x', which='both', top=False, bottom=False, left=False, right=False, labelsize=13)
        ax[i].tick_params(axis='both', which='major', labelsize=13)
        ax[i].tick_params(axis='both', which='minor', labelsize=13)
        if i != 0:
            ax[i].tick_params(labelcolor='none', axis = 'y', which='both', top=False, bottom=False, left=False, right=False, labelsize=13)
        i += 1
    fig.legend(
        [bplot["boxes"][j] for j in range(len(bplot["boxes"]))], 
        [label_map[model] for model in models],
        # bbox_to_anchor=(1.10, 0.25),
        bbox_to_anchor=(1.27, 0.25),
        loc='lower right',
        fontsize=15
    )
    plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
    # plt.xlabel(x_axis_name, fontsize = 15, labelpad=15)
    plt.ylabel('Competitive Ratio', fontsize = 20, labelpad=15)
    plt.savefig(f"data/small_box_plots.pdf", dpi=300, bbox_inches = "tight")
    plt.show()


def graph_config_to_string(config):
    graph_type = config['graph_type']
    if graph_type == 'ER':
        return f"ER_{config['p']}"
    if graph_type == 'BA':
        return f"BA_{config['ba_param']}"
    if graph_type == 'GEOM':
        return f"GEOM_{config['q']}"
    if graph_type == 'GM':
        return "GM"
    if graph_type == 'OSMNX':
        return f"OSMNX_{config['location']}"

def save_meta_experiment(graph_str, data):
    with open(f"experiment_output/meta_{graph_str}.pickle", 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

def upload_meta_experiment(graph_str, data):
    filepath = f"experiment_output/meta_{graph_str}.pickle"
    try:
        with open(filepath, 'rb') as handle:
            current_data = pickle.load(handle)
        
        for model in current_data.keys():
            if model == 'num_trials':
                current_data[model] += data[model]
            else:
                for i in range(len(current_data[model])):
                    current_data[model][i].extend(data[model][i])

        with open(filepath, 'wb') as handle:
            pickle.dump(current_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    except:
        with open(filepath, 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_meta_experiments(configs):
    data = {}
    for config in configs:
        config_str = graph_config_to_string(config)
        with open(f"experiment_output/meta_{config_str}.pickle", 'rb') as handle:
            data[config_str] = pickle.load(handle)
    return data


def _plot_meta_ratios_main(
        ratios,
        data,
        models,
        naming_function = lambda graph_type: graph_type,
        x_axis_name = "$\\vert R \\vert \; / \; \\vert L \\vert$ regime",
        confidence = 0.95,
        
    ):
    num_subplots = len(data.keys())
    # fig, ax = plt.subplots(1, num_subplots, sharex=True, sharey=True, figsize=(12,3))
    fig, ax = plt.subplots(1, num_subplots, sharex=True, sharey=True, figsize=(6,3))
    fig.add_subplot(111, frameon=False)
    i = 0
    for graph_type, graph_data in data.items():
        avg_ratios = {}

        for model, cr_by_ratio in graph_data.items():
            if model != 'num_trials':
                avg_ratios[model] = []
                for raw_crs in cr_by_ratio:
                    mean = np.nanmean(raw_crs)
                    ci_lb, ci_ub = st.norm.interval(
                        alpha=0.95, 
                        loc=mean, 
                        scale=st.sem(raw_crs, nan_policy='omit')
                    )
                    avg_ratios[model].append((mean, ci_lb, ci_ub))

        for model, model_ratios in avg_ratios.items():
            if model in models:
                competitive_ratios = [val[0] for val in model_ratios]
                ci_lbs = [val[1] for val in model_ratios]
                ci_ubs = [val[2] for val in model_ratios]
                ax[i].plot(ratios, competitive_ratios, label=label_map[model], color=color_map[model])
                ax[i].fill_between(ratios, ci_lbs, ci_ubs, alpha = 0.2, color=color_map[model])
                ax[i].grid(visible=True, which='both', axis='both')
                ax[i].set_title(title_of_graph_type2(graph_type), fontsize=17)
                ax[i].tick_params(axis='both', which='major', labelsize=13)
                ax[i].tick_params(axis='both', which='minor', labelsize=13)

                ax[i].xaxis.set_major_locator(MultipleLocator(0.5))

                handles, labels = ax[i].get_legend_handles_labels()

        i += 1

    fig.legend(
        handles,
        labels,
        # bbox_to_anchor=(1.22, 0.5),
        bbox_to_anchor=(1.38, 0.5),
        loc='right',
        fontsize=20
    )
    plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
    plt.xlabel(x_axis_name, fontsize = 20, labelpad=15)
    plt.ylabel('Average competitive ratio', fontsize = 20, labelpad=15)
    plt.savefig(f"plots/small_meta.pdf", dpi=300, bbox_inches = "tight")
    plt.show()


def _plot_meta_ratios(
        ratios,
        data,
        models,
        naming_function = lambda graph_type: graph_type,
        x_axis_name = "$\\vert R \\vert \; / \; \\vert L \\vert$ regime",
        confidence = 0.95,
        
    ):
    fig, ax = plt.subplots(4, 3, sharex=True, sharey=True, figsize=(12,16))
    fig.add_subplot(111, frameon=False)
    i = 0
    j = 0
    for graph_type, graph_data in data.items():
        avg_ratios = {}

        for model, cr_by_ratio in graph_data.items():
            if model != 'num_trials':
                avg_ratios[model] = []
                for raw_crs in cr_by_ratio:
                    mean = np.nanmean(raw_crs)
                    ci_lb, ci_ub = st.norm.interval(
                        alpha=0.95, 
                        loc=mean, 
                        scale=st.sem(raw_crs, nan_policy='omit')
                    )
                    avg_ratios[model].append((mean, ci_lb, ci_ub))

        for model, model_ratios in avg_ratios.items():
            if model in models:
                competitive_ratios = [val[0] for val in model_ratios]
                ci_lbs = [val[1] for val in model_ratios]
                ci_ubs = [val[2] for val in model_ratios]
                ax[i, j].plot(ratios, competitive_ratios, label=label_map[model], color=color_map[model])
                ax[i, j].fill_between(ratios, ci_lbs, ci_ubs, alpha = 0.2, color=color_map[model])
                ax[i, j].grid(visible=True, which='both', axis='both')
                ax[i, j].set_title(title_of_graph_type2(graph_type), fontsize=17)
                ax[i, j].tick_params(axis='both', which='major', labelsize=13)
                ax[i, j].tick_params(axis='both', which='minor', labelsize=13)
                handles, labels = ax[i, j].get_legend_handles_labels()

        j += 1
        if j == 3:
            i += 1
            j = 0

    fig.legend(
        handles,
        labels,
        bbox_to_anchor=(1.22, 0.5),
        loc='right',
        fontsize=20
    )
    plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
    plt.xlabel(x_axis_name, fontsize = 20, labelpad=15)
    plt.ylabel('Average competitive ratio', fontsize = 20, labelpad=15)
    plt.savefig(f"plots/meta_complete_plots.pdf", dpi=300, bbox_inches = "tight")
    plt.show()







