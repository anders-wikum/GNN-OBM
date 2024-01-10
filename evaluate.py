import torch
import torch_converter as pc
import algorithms as dp
import numpy as np
from torch_geometric.loader import DataLoader
from util import Dataset, diff, fill_list, _flip_coins, _vtg_greedy_choices
import torch.nn.functional as F
from algorithms import cache_stochastic_opt

from typing import Optional
from numpy.random import Generator
from typing import List, Tuple
from params import _Array, _Instance

_ModelIndex = Tuple[int, int]
BASELINES = {
    'greedy': dp.greedy,
    'lp_rounding': dp.lp_approx
}


class StateRealization:
    def __init__(self, instance: _Instance, rng: Generator, base_models: List[torch.nn.Module]):
        (A, p) = instance
        self.value = 0
        self.matching = []
        self.offline_nodes = frozenset(np.arange(A.shape[1]))
        self.dataset = pc.init_pyg(instance, rng, base_models)
        self.coin_flips = _flip_coins(p, rng)
        self.base_models = base_models

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
                self.offline_nodes,
                self.base_models
            )


class ExecutionState:
    def __init__(
        self,
        instance: _Instance,
        num_realizations: int,
        rng: Generator,
        base_models: List[torch.nn.Module]
    ):
        (A, p) = instance
        self.A = A
        self.size = A.shape
        self.p = p

        self.state_realizations = [
            StateRealization(instance, rng, base_models)
            for _ in range(num_realizations)
        ]


class ParallelExecutionState:
    def __init__(
        self,
        instances: List[_Instance],
        num_realizations: int,
        rng: Generator,
        base_models: List[torch.nn.Module]
    ):
        self.num_instances = len(instances)
        self.num_realizations = num_realizations
        self.max_online_nodes = np.max([A.shape[0] for (A, _) in instances])
        self.execution_states = [
            ExecutionState(instance, num_realizations, rng, base_models)
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
            shuffle=False
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
            shuffle=False
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


    def compute_competitive_ratios(self, baselines: List[str], **kwargs):
        ratios = {
            "learned": np.zeros(
                shape=(self.num_instances, self.num_realizations)
            )
        }

        for baseline in baselines:
            ratios[baseline] = np.zeros(
                shape=(self.num_instances, self.num_realizations)
            )

        for i, ex_state in enumerate(self.execution_states):
            A = ex_state.A
            p = ex_state.p
            for j, real_state in enumerate(ex_state.state_realizations):
                coin_flips = real_state.coin_flips
                OPT = dp.offline_opt(A, coin_flips)[1]
                if OPT > 0:
                    ratios["learned"][i, j] = real_state.value / OPT
                    for baseline in baselines:
                        ratios[baseline][i, j] = \
                            BASELINES[baseline]((A, p), real_state.coin_flips, **kwargs)[1] / OPT
                        
                else:
                    ratios["learned"][i, j] = np.nan
                    for baseline in baselines:
                        ratios[baseline] = np.nan

        return {
            name: np.nanmean(ratio_matrix, axis=1)
            for (name, ratio_matrix) in ratios.items()
        }


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

def _batch_select_test(batches):
    choices = []
    for batch in batches:
        vtg_sum = batch.base_model_preds
        vtg_sum = vtg_sum / torch.linalg.vector_norm(vtg_sum, dim=0, ord=1)
        vtg_sum = vtg_sum.sum(dim=1)
        vtg_sum = vtg_sum.to('cuda:2')
        choices.append(_vtg_greedy_choices(vtg_sum, batch))
    return torch.cat(choices)


def _compute_avg_predictions(
    base_models: List[torch.nn.Module],
    arrival_indices,
    parallel_state: ParallelExecutionState,
    batch_size: int
) -> List[torch.Tensor]:

    choices = []
    for j, _ in enumerate(base_models):
        model_arrival_indices = arrival_indices[j]
        
        if len(model_arrival_indices) == 0:
            model_choices = None
        else:
            model_batches = parallel_state.batch_data(
                model_arrival_indices,
                batch_size
            )
            model_choices = _batch_select_test(model_batches)

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
    num_realizations: Optional[int] = 1,
    baselines: Optional[List[str]] = [],
    **kwargs
) -> tuple:
    
    # ==================== State generation =============================== #
    
    parallel_state = ParallelExecutionState(
        instances,
        num_realizations,
        rng,
        base_models
    )

    # ===================================================================== #

    num_models = len(base_models)
    model_assignment = None

    for t in range(parallel_state.max_online_nodes):

        model_assignment = parallel_state._model_assign(
            meta_model,
            meta_model_type,
            batch_size
        )

        non_arrival_indices, arrival_indices = parallel_state.get_arrivals(
            t,
            model_assignment,
            num_models
        )
        
        # choices = _compute_base_model_predictions(
        #     base_models,
        #     arrival_indices,
        #     parallel_state,
        #     batch_size
        # )

        choices = _compute_avg_predictions(
            base_models,
            arrival_indices,
            parallel_state,
            batch_size
        )

        parallel_state.update(t, choices, non_arrival_indices, arrival_indices)

    ratio_dict = parallel_state.compute_competitive_ratios(
        baselines,
        **kwargs
    )

    return ratio_dict


def pp_output(ratios: dict) -> None:
    print('-- Competitive ratios --')
    for (name, approx_ratios) in ratios.items():
        print(f"{name}: {np.mean(approx_ratios).round(4)}")
    
