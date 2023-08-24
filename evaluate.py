import torch
import math
import time
import torch_converter as pc
import algorithms as dp
import numpy as np
from torch_geometric.loader import DataLoader
from util import Dataset, diff, objectview, fill_list

from typing import Optional
from numpy.random import Generator
from typing import List
from params import _Array, _Instance




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
            pc._update_pyg(
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
    'classification': _skip_class,
    'meta': _vtg_greedy
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
def evaluate_model(
    model: object,
    args: dict,
    instances: List[_Instance],
    coin_flips: _Array,
    batch_size: int,
    rng: Generator
) -> tuple:
    
    m, n = instances[0][0].shape
    if type(args) is dict:
        args = objectview(args)
    log = _init_log()
    device = args.device
    head = args.head

    
    # ======================= Data generation ============================= #
    pre_data_gen = time.perf_counter()  

    state = pc._instances_to_gnn_eval_state(
        instances,
        coin_flips,
        rng
    )
    
    post_data_gen = time.perf_counter()
    _update_log(log, 'Data generation', post_data_gen - pre_data_gen)
    # ======================== END ======================================== #
    

    for t in range(m):
        arrivals = state['coin_flips'][t]
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

                    pred = model(batch)

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


def _select_eval_model(classify_model, data) -> object:
    with torch.no_grad():
        y = classify_model(data)
        return torch.argmax(y, dim=1)


def _meta_batch_data(state, arrival_indices, batch_size):
    iter_data = Dataset(
        [
            state['dataset'][i]
            for i in arrival_indices
        ]
    )

    iter_loader = DataLoader(
        iter_data,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True
    )

    return iter_loader


def _meta_update_state(
    choices: List[torch.Tensor],
    state: dict,
    arrivals: list,
    model_index: int,
    t: int,
    m: int,
    n: int
) -> None:
    arrival_index = 0
    if choices is not None:
        choices = choices.cpu()

    for i in state['model_assign'][model_index]:
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
            pc._update_pyg(
                state['dataset'][i],
                t + 1,
                choice,
                state['As'][i],
                state['offline_nodes'][i]
            )


def evaluate_meta_model(
    classify_model: object,
    eval_models: List[object],
    args: dict,
    instances: List[_Instance],
    coin_flips: _Array,
    batch_size: int,
    rng: Generator
) -> tuple:
    
    m, n = instances[0][0].shape
    if type(args) is dict:
        args = objectview(args)
    log = _init_log()
    device = args.device
    head = args.head

    
    # ======================= Data generation ============================= #
    pre_data_gen = time.perf_counter()  

    state = pc._instances_to_gnn_eval_state(
        instances,
        coin_flips,
        rng
    )

    num_models = len(eval_models)
    num_instances = len(instances)

    X = pc._instances_to_nn_eval(instances).to(device)
    model_indices = _select_eval_model(classify_model, (X, None))
    state['model_assign'] = fill_list([], num_models + 1)
    for i in range(num_instances):
        state['model_assign'][model_indices[i].item()].append(i)

    post_data_gen = time.perf_counter()
    _update_log(log, 'Data generation', post_data_gen - pre_data_gen)
    # ======================== END ======================================== #


    
    
    for t in range(m):
        arrivals = state['coin_flips'][t]
        arrival_indices = fill_list([], len(eval_models))

        for j, model in enumerate(eval_models):
            arrival_indices = [
                i 
                for i in state['model_assign'][j]
                if arrivals[i]
            ]
            num_arrivals = len(arrival_indices)
            if num_arrivals > 0:
                
                # ======================= Batching ============================ #
                pre_batch = time.perf_counter()

                batches = _meta_batch_data(state, arrival_indices, batch_size)

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

                        pred = model(batch)

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

            _meta_update_state(choices, state, arrivals, j, t, m, n)

            post_state = time.perf_counter()
            _update_log(log, 'State update', post_state - pre_state)
            # ======================== END ==================================== #

    for i in state['model_assign'][num_models]:
        matching, value = dp.greedy(
            A=state['As'][i],
            coin_flips=state['coin_flips'][:, i],
            r=0
        )
        state['matchings'][i] = matching
        state['values'][i] = value

    return _compute_competitive_ratios(state), log


def pp_output(ratios: list, log: dict, show_log: Optional[bool] = False) -> None:
    if show_log:
        print('-- Execution time --')
        for key, val in log.items():
            print(f"{key}: {np.mean(val).round(4)} sec")

        print()
    print('-- Competitive ratios --')
    print(f"GNN: {np.mean(ratios[0]).round(4)}")
    print(f"Greedy: {np.mean(ratios[1]).round(4)}")
