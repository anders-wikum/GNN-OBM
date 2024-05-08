import torch
import torch_converter as pc
import algorithms as dp
import numpy as np
from torch_geometric.loader import DataLoader
from util import Dataset, diff, fill_list, _flip_coins, _vtg_greedy_choices
from typing import Optional
from numpy.random import Generator
from typing import List, Tuple
from params import _Array, _Instance
import time

_ModelIndex = Tuple[int, int]
BASELINES = {
    'greedy': dp.greedy,
    'greedy_t': dp.threshold_greedy,
    'lp_rounding': dp.lp_approx,
    'naor_lp_rounding': dp.naor_lp_approx
}


class StateRealization:
    def __init__(self, instance: _Instance, rng: Generator, base_models: List[torch.nn.Module]):
        (A, p, _, _) = instance
        self.value = 0
        self.matching = []
        self.offline_nodes = frozenset(np.arange(A.shape[1]))
        self.dataset = pc.init_pyg(instance, rng, base_models)
        self.coin_flips = _flip_coins(p, rng)
        self.base_models = base_models

    def update(self, instance, choice: int, t: int):
        A, _, _, _ = instance
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
        (A, p, noisy_A, noisy_p) = instance
        self.A = A
        self.size = A.shape
        self.p = p
        self.noisy_A = noisy_A
        self.noisy_p = noisy_p


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
        self.instances = instances
        self.num_realizations = num_realizations
        self.max_online_nodes = np.max([A.shape[0] for (A, _, _, _) in instances])
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
            return np.zeros(online_offline_ratios.shape[0]).astype(int)
        return (online_offline_ratios > 1.5).astype(int)


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
            
    def _model_assign(
        self,
        meta_model: object,
        meta_model_type: str,
        batch_size: Optional[int] = None
    ):
        if meta_model_type == 'gnn':
            return self._gnn_model_assign(meta_model, batch_size)
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
                    instance = (self.execution_states[i].A, self.execution_states[i].p, self.execution_states[i].noisy_A, self.execution_states[i].noisy_p)
                    state = self.execution_states[i].state_realizations[j]
                    state.update(instance, choice, t)
                    arrival_index += 1
        
        for (i, j) in non_arrival_indices:
            state = self.execution_states[i].state_realizations[j]
            instance = (self.execution_states[i].A, self.execution_states[i].p, self.execution_states[i].noisy_A, self.execution_states[i].noisy_p)
            state.update(instance, -1, t)


    def compute_competitive_ratios(self, baselines: List[str], **kwargs):
        m = self.execution_states[0].A.shape[0]

        ratios = {
            "learned": np.zeros(
                shape=(self.num_instances, self.num_realizations)
            )
        }
        inputs = [
            (instance[2], instance[3])
            for instance in self.instances
        ]
        lp_outputs = dp.call_model(inputs)
        times = {baseline: 0 for baseline in baselines}


        for baseline in baselines:
            ratios[baseline] = np.zeros(
                shape=(self.num_instances, self.num_realizations)
            )
        
        for i, ex_state in enumerate(self.execution_states):
            A = ex_state.A
            p = ex_state.p
            noisy_A = ex_state.noisy_A
            noisy_p = ex_state.noisy_p
            kwargs["lp_rounding"]["x"] = lp_outputs[i]
            kwargs["naor_lp_rounding"]["x"] = lp_outputs[i]
            
            for j, real_state in enumerate(ex_state.state_realizations):
                coin_flips = real_state.coin_flips
                _, OPT = dp.offline_opt(A, coin_flips)

                if OPT > 0:
                    ratios["learned"][i, j] = real_state.value / OPT
                    for baseline in baselines:
                        base_start = time.perf_counter()
                        _, value = BASELINES[baseline](
                            (A, p, noisy_A, noisy_p),
                            real_state.coin_flips,
                            **kwargs[baseline]
                        )
                        ratios[baseline][i, j] = value / OPT
                        base_end = time.perf_counter()
                        times[baseline] += base_end - base_start
           
                        
                else:
                    ratios["learned"][i, j] = np.nan
                    for baseline in baselines:
                        ratios[baseline][i, j] = np.nan
                        
        avg_ratios = {
            name: list(np.nanmean(ratio_matrix, axis=1))
            for (name, ratio_matrix) in ratios.items()
        }

        return avg_ratios, times


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
    
    gen_start = time.perf_counter()
    parallel_state = ParallelExecutionState(
        instances,
        num_realizations,
        rng,
        base_models
    )
    gen_end = time.perf_counter()

    # ===================================================================== #

    num_models = len(base_models)
    model_assignment = None

    gnn_start = time.perf_counter()
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
        
        choices = _compute_base_model_predictions(
            base_models,
            arrival_indices,
            parallel_state,
            batch_size
        )

        parallel_state.update(t, choices, non_arrival_indices, arrival_indices)

    gnn_end = time.perf_counter()


    ratio_dict, times = parallel_state.compute_competitive_ratios(
        baselines,
        **kwargs
    )

    print(f"Generation time: {gen_end - gen_start}")
    print(f"GNN time: {gnn_end - gnn_start}")
    print(f"Baseline times: {times}")

    return ratio_dict, {}


def pp_output(ratios: dict) -> None:
    print('-- Competitive ratios --')
    for (name, approx_ratios) in ratios.items():
        print(f"{name}: {np.mean(approx_ratios).round(4)}")
    
