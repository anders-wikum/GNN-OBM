import torch
import math
import time
import torch_converter as pc
import algorithms as dp
import numpy as np
from torch_geometric.loader import DataLoader
from util import Dataset, diff, objectview, fill_list, _flip_coins

from typing import Optional
from numpy.random import Generator
from typing import List, Tuple
from params import _Array, _Instance

_ModelIndex = Tuple[int, int]

class StateRealization:
    def __init__(self, instance: _Instance, rng: Generator):
        (A, p) = instance
        self.value = 0
        self.matching = []
        self.offline_nodes = frozenset(np.arange(A.shape[1]))
        self.dataset = pc._to_pyg(A, p, self.offline_nodes, 0, rng)
        self.coin_flips = _flip_coins(p, rng)

    def update(self, choice: int, t: int, A: _Array):

        # If we don't skip, update state
        if choice != -1:
            self.matching.append((t, choice))
            self.value += A[t, choice]
            self.offline_nodes = diff(self.offline_nodes, choice)

        # If still in a relevant timestep, update dataset
        if t < A.shape[0] - 1:
            pc._update_pyg(
                self.dataset,
                t + 1,
                choice,
                A,
                self.offline_nodes
            )


class ExecutionState:
    def __init__(
        self,
        instance: _Instance,
        num_realizations: int,
        rng: Generator
    ):
        (A, p) = instance
        self.A = A
        self.size = A.shape
        self.p = p

        self.state_realizations = [
            StateRealization(instance, rng)
            for _ in range(num_realizations)
        ]


class ParallelExecutionState:
    def __init__(
        self,
        instances: List[_Instance],
        num_realizations: int,
        rng: Generator
    ):
        self.num_instances = len(instances)
        self.num_realizations = num_realizations
        self.max_online_nodes = np.max([A.shape[0] for (A, _) in instances])
        self.execution_states = [
            ExecutionState(instance, num_realizations, rng)
            for instance in instances
        ]


    def get_arrivals(
        self,
        t: int,
        model_assignment: _Array,
        num_models: int,
    ) -> Tuple[List[_ModelIndex], List[List[_ModelIndex]]]:

        non_arrival_indices = []
        arrival_indices = fill_list([], num_models)

        for i, ex_state in enumerate(self.execution_states):
            for j, real_state in enumerate(ex_state.state_realizations):
                if (t < len(ex_state.p)) and real_state.coin_flips[t]:
                    arrival_indices[model_assignment[i]].append((i, j))
                else:
                    non_arrival_indices.append((i, j))

        return non_arrival_indices, arrival_indices
    

    def batch_data(
        self,
        model_arrival_indices: List[_ModelIndex],
        batch_size: int
    ) -> DataLoader:
        
        data = Dataset(
            [
                self.execution_states[i].state_realizations[j].dataset
                for (i, j) in model_arrival_indices
            ]
        )

        return DataLoader(
            data,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True
        )
    

    def model_assign(
        self,
        classify_model: object,
        device: str
    ):
        instances = [
            (execution_state.A, execution_state.p)
            for execution_state in self.execution_states
        ]

        if classify_model is None:
            return np.zeros(len(instances)).astype(int)
        
        else:
            X = pc._instances_to_nn_eval(instances).to(device)
            with torch.no_grad():
                return _select_eval_model(classify_model, (X, None)) \
                    .cpu() \
                    .numpy() \
                    .astype(int)
            

    def update(
        self,
        t: int,
        choices: List[torch.Tensor],
        non_arrival_indices: List[_ModelIndex],
        arrival_indices: List[List[_ModelIndex]]
    ):
        for model_index, model_arrivals in enumerate(arrival_indices[:-1]):
            if choices[model_index] is not None:
                arrival_index = 0
                for (i, j) in model_arrivals:
                    choice = choices[model_index][arrival_index].item()
                    A = self.execution_states[i].A
                    state = self.execution_states[i].state_realizations[j]
                    state.update(choice, t, A)
                    arrival_index += 1
        
        for (i, j) in non_arrival_indices:
            state = self.execution_states[i].state_realizations[j]
            A = self.execution_states[i].A
            state.update(-1, t, A)


    def compute_competitive_ratios(self):
        learned_ratios = np.zeros(
            shape=(self.num_instances, self.num_realizations)
        )
        greedy_ratios = np.zeros(
            shape=(self.num_instances, self.num_realizations)
        )

        for i, ex_state in enumerate(self.execution_states):
            A = ex_state.A
            for j, real_state in enumerate(ex_state.state_realizations):
                OPT = dp.offline_opt(A, real_state.coin_flips)[1]
                if OPT > 0:
                    learned_ratios[i, j] = real_state.value / OPT
                    greedy_ratios[i, j] = \
                        dp.greedy(A, real_state.coin_flips, 0)[1] / OPT
                else:
                    learned_ratios[i, j] = np.nan
                    greedy_ratios[i, j] = np.nan

        return (
            np.nanmean(learned_ratios, axis=1),
            np.nanmean(greedy_ratios, axis=1),
        )


def _select_eval_model(classify_model, data) -> object:
    with torch.no_grad():
        y = classify_model(data)
        return torch.argmax(y, dim=1)
    

def _masked_argmax(tensor, mask, dim):
    masked = torch.mul(tensor, mask)
    neg_inf = torch.zeros_like(tensor)
    neg_inf[~mask] = -math.inf 
    return (masked + neg_inf).argmax(dim=dim)


def _vtg_greedy(
    pred: torch.Tensor,
    batch: Dataset,
    **kwargs
) -> torch.Tensor:
    
    batch_size = batch.ptr.size(dim=0) - 1
    choices = _masked_argmax(
                pred.view(batch_size, -1),
                batch.neighbors.view(batch_size, -1),
                dim=1
            )
    
    return torch.where(choices < batch.n, choices, -1)


EVALUATORS = {
    'regression': _vtg_greedy,
    'meta': _vtg_greedy
}


def _compute_eval_model_predictions(
    eval_models: List[object],
    arrival_indices,
    parallel_state: ParallelExecutionState,
    batch_size: int,
    args: dict
) -> List[torch.Tensor]:
    
    choices = []
    for j, model in enumerate(eval_models):
        model_arrival_indices = arrival_indices[j]
        
        if len(model_arrival_indices) == 0:
            model_choices = None
        else:
            model_batches = parallel_state.batch_data(
                model_arrival_indices,
                batch_size
            )

            with torch.no_grad():
                model_choices = []
                for batch in model_batches:
                    batch.to(args.device)
                    pred = model(batch)
                    model_choices.append(
                        EVALUATORS[args.head](
                            pred=pred,
                            batch=batch
                        )
                    )
                model_choices = torch.cat(model_choices)
                #print(j, model_choices)
        choices.append(model_choices)
    

    return choices

def _execute_greedy(
    parallel_state: ParallelExecutionState,
    model_assignment: _Array,
    num_models: int
) -> None:
    for i, ex_state in enumerate(parallel_state.execution_states):
        if model_assignment[i] == num_models - 1:
            for real_state in ex_state.state_realizations:
                real_state.matching, real_state.value = dp.greedy(
                    A=ex_state.A,
                    coin_flips=real_state.coin_flips,
                    r=0
                )



def evaluate_model(
    classify_model: object,
    eval_models: List[object],
    args: dict,
    instances: List[_Instance],
    batch_size: int,
    rng: Generator,
    num_realizations: Optional[int] = 1
) -> tuple:
    
    # ==================== State generation =============================== #
    
    parallel_state = ParallelExecutionState(
        instances,
        num_realizations,
        rng
    )

    # ===================================================================== #
    if type(args) is dict:
        args = objectview(args)
    device = args.device
    num_models = len(eval_models) + 1

    model_assignment = parallel_state.model_assign(
        classify_model,
        device
    )

    for t in range(parallel_state.max_online_nodes):
        non_arrival_indices, arrival_indices = parallel_state.get_arrivals(
            t,
            model_assignment,
            num_models
        )

        choices = _compute_eval_model_predictions(
            eval_models,
            arrival_indices,
            parallel_state,
            batch_size,
            args
        )

        parallel_state.update(t, choices, non_arrival_indices, arrival_indices)

    _execute_greedy(parallel_state, model_assignment, num_models)
    learned_ratios, greedy_ratios = parallel_state.compute_competitive_ratios()
    return (learned_ratios, greedy_ratios)


def pp_output(ratios: list, log: dict, show_log: Optional[bool] = False) -> None:
    if show_log:
        print('-- Execution time --')
        for key, val in log.items():
            print(f"{key}: {np.mean(val).round(4)} sec")

        print()
    print('-- Competitive ratios --')
    print(f"GNN: {np.mean(ratios[0]).round(4)}")
    print(f"Greedy: {np.mean(ratios[1]).round(4)}")
