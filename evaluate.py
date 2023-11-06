import torch
import torch_converter as pc
import algorithms as dp
import numpy as np
from torch_geometric.loader import DataLoader
from util import Dataset, diff, fill_list, _flip_coins
import torch.nn.functional as F
from algorithms import cache_stochastic_opt

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
        self.dataset = pc.init_pyg(instance, rng)
        self.coin_flips = _flip_coins(p, rng)

    def update(self, instance, choice: int, t: int):
        A, _ = instance
        # If we don't skip, update state
        if choice != -1:
            self.matching.append((t, choice))
            self.value += A[t, choice]
            self.offline_nodes = diff(self.offline_nodes, choice)

        # If still in a relevant timestep, update dataset
        if t < A.shape[0] - 1:
            pc.update_pyg(
                self.dataset,
                instance,
                choice,
                t + 1,
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
    
    def _heuristic_model_assign(self, meta_model):
        online_offline_ratios = np.array([
            realized_state.dataset.graph_features[1].item()
            for execution_state in self.execution_states
            for realized_state in execution_state.state_realizations
        ])
        if meta_model is None:
            return np.zeros(len(online_offline_ratios)).astype(int)
        return (online_offline_ratios < 1.5).astype(int)
    

    def _shakey_model_assign(self, base_models: List[object], epsilon: float):
        assignment = np.zeros(self.num_instances)
        coin_flips = np.random.binomial(1, epsilon, self.num_instances)
        non_opt_model_indices = np.arange(1, len(base_models) + 1)
        for i, _ in enumerate(assignment):
            if coin_flips[i]:
                assignment[i] = np.random.choice(non_opt_model_indices)
        return assignment.astype(int)

    
    def _hybrid_model_assign(
        self,
        meta_model: object,
        batch_size: int
    ):
        
        assignment_len = self.num_instances * self.num_realizations
        if meta_model is None:
            return np.zeros(assignment_len).astype(int)

        device = next(meta_model.parameters()).device
        data_list = [
            realized_state.dataset
            for execution_state in self.execution_states
            for realized_state in execution_state.state_realizations
        ]
        gnn1_mask = [False] * assignment_len
        gnn2_mask = [False] * assignment_len
        meta_learner_mask = [False] * assignment_len

        for i, data in enumerate(data_list):
            online_offline_ratio = data.graph_features[1].item()

            if online_offline_ratio >= 2.0:
                gnn1_mask[i] = True
            elif data.graph_features[1].item() < 0.75:
                gnn2_mask[i] = True
            else:
                meta_learner_mask[i] = True

        meta_learner_data_list = [
            data_list[i] for i, boolean in enumerate(meta_learner_mask)
            if boolean
        ]
  
        assignment = np.zeros(assignment_len)
        assignment[gnn1_mask] = 0
        assignment[gnn2_mask] = 1
        
        if len(meta_learner_data_list) > 0:
            data_loader = DataLoader(
                Dataset(meta_learner_data_list),
                batch_size=batch_size,
                shuffle=False,
                pin_memory=True
            )

            with torch.no_grad():
                preds = []
                for batch in data_loader:
                    batch.to(device)
                    pred = _select_base_model(meta_model, batch)
                    preds.append(pred)
                preds = torch.cat(preds) \
                    .round() \
                    .cpu() \
                    .numpy()
        
            assignment[meta_learner_mask] = preds

        return assignment.astype(int)


    def _gnn_model_assign(
        self,
        meta_model: object,
        batch_size: int
    ):
        
        device = next(meta_model.parameters()).device
        data = Dataset([
            realized_state.dataset
            for execution_state in self.execution_states
            for realized_state in execution_state.state_realizations
        ])

        if meta_model is None:
            return np.zeros(self.num_instances * self.num_realizations).astype(int)

        data_loader = DataLoader(
            data,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True
        )

        with torch.no_grad():
            preds = []
            for batch in data_loader:
                batch.to(device)
                pred = _select_base_model(meta_model, batch)
                preds.append(pred)
            preds = torch.cat(preds)
        return torch.round(preds) \
            .cpu() \
            .numpy() \
            .astype(int)



    def _nn_model_assign(
        self,
        meta_model: object
    ):
        
        device = next(meta_model.parameters()).device
        instances = [
            (execution_state.A, execution_state.p)
            for execution_state in self.execution_states
        ]

        if meta_model is None:
            return np.zeros(len(instances)).astype(int)
        
        else:
            X = pc._instances_to_nn_eval(instances).to(device)
            with torch.no_grad():
                return _select_base_model(meta_model, (X, None)) \
                    .cpu() \
                    .numpy() \
                    .astype(int)
            
    def _model_assign(
        self,
        meta_model: object,
        meta_model_type: str,
        batch_size: Optional[int] = None
    ):
        if meta_model_type == 'gnn':
            return self._gnn_model_assign(meta_model, batch_size)
        elif meta_model_type == 'hybrid':
            return self._hybrid_model_assign(meta_model, batch_size)
        elif meta_model_type == 'nn':
            return self._nn_model_assign(meta_model)
        else:
            return self._heuristic_model_assign(meta_model)
            

    def update(
        self,
        t: int,
        choices: List[torch.Tensor],
        non_arrival_indices: List[_ModelIndex],
        arrival_indices: List[List[_ModelIndex]]
    ):
        for model_index, model_arrivals in enumerate(arrival_indices):
            if choices[model_index] is not None:
                arrival_index = 0
                for (i, j) in model_arrivals:
                    choice = choices[model_index][arrival_index].item()
                    instance = (self.execution_states[i].A, self.execution_states[i].p)
                    state = self.execution_states[i].state_realizations[j]
                    state.update(instance, choice, t)
                    arrival_index += 1
        
        for (i, j) in non_arrival_indices:
            state = self.execution_states[i].state_realizations[j]
            instance = (self.execution_states[i].A, self.execution_states[i].p)
            state.update(instance, -1, t)


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


def _select_base_model(meta_model, data) -> object:
    with torch.no_grad():
        y = meta_model(data)
        return torch.argmax(y, dim=1)
    

def _compute_base_model_predictions(
    base_models: List[torch.nn.Module],
    arrival_indices,
    parallel_state: ParallelExecutionState,
    batch_size: int
) -> List[torch.Tensor]:

    choices = []
    for j, model in enumerate(base_models):
        model_arrival_indices = arrival_indices[j]
        
        if len(model_arrival_indices) == 0:
            model_choices = None
        else:
            model_batches = parallel_state.batch_data(
                model_arrival_indices,
                batch_size
            )
            model_choices = model.batch_select_match_nodes(model_batches)

        choices.append(model_choices)

    return choices

def _compute_shakey_model_predictions(
    base_models: List[torch.nn.Module],
    caches: List[dict],
    arrival_indices,
    parallel_state: ParallelExecutionState,
    batch_size: int,
    t: int
) -> List[torch.Tensor]:

    choices = []

    opt_choices = []
    for (i, j) in arrival_indices[0]:
        offline_nodes = parallel_state.execution_states[i].state_realizations[j].offline_nodes
        choice = caches[i][t][offline_nodes][1]
        if choice not in offline_nodes:
            choice = -1
        opt_choices.append(choice)

    choices.append(torch.tensor(opt_choices))


    for j, model in enumerate(base_models):
        model_arrival_indices = arrival_indices[j + 1]
        
        if len(model_arrival_indices) == 0:
            model_choices = None
        else:
            model_batches = parallel_state.batch_data(
                model_arrival_indices,
                batch_size
            )
            model_choices = model.batch_select_match_nodes(model_batches)

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
    meta_model: object,
    meta_model_type: str,
    base_models: List[torch.nn.Module],
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

    num_models = len(base_models)
    model_assignment = None

    for t in range(parallel_state.max_online_nodes):
        if model_assignment is None:
            model_assignment = parallel_state._model_assign(
                meta_model,
                meta_model_type,
                batch_size
            )
            print(np.unique(model_assignment, return_counts=True))

        non_arrival_indices, arrival_indices = parallel_state.get_arrivals(
            t,
            model_assignment,
            num_models
        )
        
        choices = _compute_base_model_predictions(
            base_models,
            arrival_indices,
            parallel_state,
            batch_size
        )

        parallel_state.update(t, choices, non_arrival_indices, arrival_indices)

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


def evaluate_shakey_opt(
    epsilon: float,
    base_models: List[torch.nn.Module],
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

    caches = [
        cache_stochastic_opt(*instance)
        for instance in instances
    ]

    # ===================================================================== #

    num_models = len(base_models)
    model_assignment = None

    for t in range(parallel_state.max_online_nodes):
        model_assignment = parallel_state._shakey_model_assign(
            base_models,
            epsilon
        )

        non_arrival_indices, arrival_indices = parallel_state.get_arrivals(
            t,
            model_assignment,
            num_models + 1
        )
        
        choices = _compute_shakey_model_predictions(
            base_models,
            caches,
            arrival_indices,
            parallel_state,
            batch_size,
            t
        )

        parallel_state.update(t, choices, non_arrival_indices, arrival_indices)

    learned_ratios, greedy_ratios = parallel_state.compute_competitive_ratios()
    return (learned_ratios, greedy_ratios)
