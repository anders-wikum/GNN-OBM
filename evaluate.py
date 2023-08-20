import torch
import math
import time
import graph_generator as gg
import algorithms as dp
import numpy as np
from torch_geometric.loader import DataLoader
from util import Dataset, diff



def _masked_argmax(tensor, mask, dim):
    masked = torch.mul(tensor, mask)
    neg_inf = torch.zeros_like(tensor)
    neg_inf[~mask] = -math.inf 
    return (masked + neg_inf).argmax(dim=dim)


def _init_log():
    return {
        'Data generation': [],
        'Data batching': [],
        'Inference': [],
        'State update': []
    }


def _update_log(log, key, value):
    log[key].append(value)


def _batch_data(state, arrivals, batch_size):
    num_trials = len(state['As'])
    iter_data = Dataset(
        [state['dataset'][i] for i in range(num_trials) if arrivals[i]]
    )

    iter_loader = DataLoader(
        iter_data,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True
    )

    return iter_loader


def _func_fill_list(func: callable, size: int, **kwargs):
    return [
        func(i, **kwargs)
        for i in range(size)
    ]


def _init_state(m: int, n: int, num_trials: int, config: dict) -> dict:
    state = {
        'offline_nodes': _func_fill_list(
            lambda _: frozenset(np.arange(n)),
            num_trials
        ),
        'matchings': _func_fill_list(
            lambda _: [],
            num_trials
        ),
        'values': _func_fill_list(
            lambda _: 0,
            num_trials
        ),
        'As': _func_fill_list(
            lambda _: gg.sample_bipartite_graph(m, n, **config),
            num_trials
        ),
    }

    p = np.random.uniform(0.5, 1, (m, num_trials))
    coin_flips = np.vectorize(lambda x: np.random.binomial(1, x))(p)
    dataset = [
        gg._to_pyg(state['As'][i], p[:, i], state['offline_nodes'][i], 0)
        for i in range(num_trials)
    ]
    state['coin_flips'] = coin_flips
    state['dataset'] = dataset

    return state


def _update_state(
    choices: torch.Tensor,
    state: dict,
    arrivals: list,
    t: int,
    m: int,
    n: int
) -> None:
    arrival_index = 0
    choices = choices.cpu()
    for i in range(len(state['As'])):
        if arrivals[i]:
            choice = choices[arrival_index].item()
            arrival_index += 1
            if choice != -1:
                state['matchings'][i].append((t, choice))
                state['values'][i] += state['As'][i][t, choice]
                state['offline_nodes'][i] = diff(state['offline_nodes'][i], choice)
        else:
            choice = -1
            
        if t < m - 1:
            gg._update_pyg(
                state['dataset'][i],
                t + 1,
                choice,
                state['As'][i],
                state['offline_nodes'][i]
            )


def _vtg_greedy(
    pred: torch.Tensor,
    batch: Dataset,
    m: int,
    n: int,
    **kwargs
) -> torch.Tensor:
    batch_size = batch.ptr.size(dim=0) - 1
    choices = _masked_argmax(
                pred.view(batch_size, -1),
                batch.neighbors.view(batch_size, -1),
                dim=1
            )
    
    return torch.where(choices < n, choices, -1)


def _skip_class(
    pred: torch.Tensor,
    batch: Dataset,
    m: int,
    n: int,
    **kwargs
) -> torch.Tensor:
    
    As = kwargs['state']['As']
    batch_num = kwargs['batch_id']
    arrivals = kwargs['arrivals']
    t = kwargs['t']
    device = kwargs['device']

    batch_size = batch.ptr.size(dim=0) - 1
    i = batch_num * kwargs['batch_size']
    all_indices = np.arange(1, arrivals.shape[0] + 1)
    arrival_indices = np.multiply(all_indices, arrivals)
    arrival_indices = arrival_indices[arrival_indices != 0][i: i + batch_size] - 1
    
    adj = torch.tensor([As[i][t, :] for i in arrival_indices]).to(device)
    greedy_choices = _masked_argmax(
        adj,
        batch.neighbors.view(batch_size, -1)[:, :n],
        dim=1
    )

    return torch.where(pred < 0.5, greedy_choices, -1)


EVALUATORS = {
    'regression': _vtg_greedy,
    'classification': _skip_class
}


def _compute_competitive_ratios(state):
    learned_ratios = []; greedy_ratios = []
    num_trials = len(state['As'])

    for i in range(num_trials):
        OPT = dp.offline_opt(state['As'][i], state['coin_flips'][:, i])[1]
        if OPT > 0:
            learned_ratios.append(state['values'][i] / OPT)
            greedy_ratios.append(
                dp.greedy(state['As'][i], state['coin_flips'][:, i], 0)[1] / OPT
            )

    return learned_ratios, greedy_ratios

#TODO Add threshold option to classification evaluator
def batched_test_model(model, args, num_trials, batch_size, m, n, config):
    log = _init_log()
    device = args.device
    head = args.head

    # ======================= Data generation ============================= #
    pre_data_gen = time.perf_counter()  

    state = _init_state(m, n, num_trials, config)
    
    post_data_gen = time.perf_counter()
    _update_log(log, 'Data generation', post_data_gen - pre_data_gen)
    # ======================== END ======================================== #
    
    print(m, n, config)
    
    for t in range(m):
        arrivals = state['coin_flips'][t, :]
        num_arrivals = int(sum(arrivals))
        if num_arrivals > 0:
            
            # ======================= Batching ============================ #
            pre_batch = time.perf_counter()

            batches = _batch_data(state, arrivals, batch_size)

            post_batch = time.perf_counter()
            _update_log(log, 'Data batching', post_batch - pre_batch)
            # ======================= END ================================= #

            # ======================= Inference =========================== #
            pre_infer = time.perf_counter()
            kwargs = {
                'state': state,
                'arrivals': arrivals,
                'batch_size': batch_size,
                't': t,
                'device': device
            }
            with torch.no_grad():
                choices = []
                for i, batch in enumerate(batches):
                    kwargs['batch_id'] = i
                    batch.to(device)
                    pred = model(
                        batch.x,
                        batch.edge_index,
                        batch.edge_attr,
                        batch.batch,
                        batch.ptr.size(dim=0) - 1, 
                        batch.graph_features
                    )
                    choices.append(
                        EVALUATORS[head](
                            pred=pred,
                            batch=batch,
                            m=m,
                            n=n,
                            **kwargs
                        )
                    )
                choices = torch.cat(choices)

            post_infer = time.perf_counter()
            _update_log(log, 'Inference', post_infer - pre_infer)
            # ======================= END ================================= #

        else:
            choices = None

        # ======================= State update ============================ #
        pre_state = time.perf_counter()

        _update_state(choices, state, arrivals, t, m, n)

        post_state = time.perf_counter()
        _update_log(log, 'State update', post_state - pre_state)
        # ======================== END ==================================== #

    return _compute_competitive_ratios(state), log