import torch
import torch_converter as pc
import algorithms as dp
import numpy as np
from torch_geometric.loader import DataLoader
from util import Dataset, diff, fill_list, _flip_coins, _masked_argmax
from typing import Optional
from numpy.random import Generator
from typing import List, Tuple
from params import _Array, _Instance
from collections import defaultdict
import time

_ModelIndex = Tuple[int, int]
BASELINES = {
    'greedy': dp.greedy,
    'greedy_t': dp.threshold_greedy,
    'lp_rounding': dp.lp_approx,
    'naor_lp_rounding': dp.naor_lp_approx
}


class StateRealization:
    def __init__(self, instance: _Instance, rng: Generator):
        (A, p, _, _) = instance
        self.value = 0
        self.matching = []
        self.offline_nodes = frozenset(np.arange(A.shape[1]))
        self.dataset = pc.init_pyg(instance, rng)
        self.coin_flips = _flip_coins(p, rng)

    def update(self, instance, choice: int, t: int):
        A, _, _, _ = instance
        # If we don't skip, update state
        
        state_start = time.perf_counter()
        if choice != -1:
            self.matching.append((t, choice))
            self.value += A[t, choice]
            self.offline_nodes = diff(self.offline_nodes, choice)
        state_end = time.perf_counter()

        # If still in a relevant timestep, update dataset
        pyg_start = time.perf_counter()
        if t < A.shape[0] - 1:
            pc.update_pyg(
                self.dataset,
                instance,
                choice,
                t + 1,
                self.offline_nodes
            )
        pyg_end = time.perf_counter()
        return state_end - state_start, pyg_end - pyg_start


class ExecutionState:
    def __init__(
        self,
        instance: _Instance,
        num_realizations: int,
        rng: Generator
    ):
        (A, p, noisy_A, noisy_p) = instance
        self.instance = instance
        self.A = A
        self.size = A.shape
        self.p = p
        self.noisy_A = noisy_A
        self.noisy_p = noisy_p


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
        self.instances = instances
        self.num_realizations = num_realizations
        self.m, self.n = self.instances[0][0].shape
        self.execution_states = [
            ExecutionState(instance, num_realizations, rng)
            for instance in instances
        ]
    

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
        base_models: List[torch.nn.Module],
        batch_size: int
    ):
        
        device = next(base_models[0].parameters()).device

        data = Dataset([
            realized_state.dataset
            for execution_state in self.execution_states
            for realized_state in execution_state.state_realizations
        ])

        if meta_model is None:
            index = np.zeros(self.num_instances * self.num_realizations).astype(int)

        data_loader = DataLoader(
            data,
            batch_size=batch_size,
            shuffle=False
        )

        with torch.no_grad():
            preds = []
            base_preds = []
            for batch in data_loader:
                batch.to(device)

                # Compute base model predictions
                base_pred = torch.concat(
                    [
                        base_model(batch)
                        for base_model in base_models
                    ],
                    dim=1
                )
        
                batch.base_model_preds = base_pred
                base_preds.append(base_pred)

                if meta_model is not None:
                    # Compute meta model predictions
                    pred = _select_base_model(meta_model, batch).int()
                    preds.append(pred)
                    index = torch.cat(preds)
            # print(f"base_preds: {base_preds[0].size(), base_preds}")
            
            base_preds = torch.cat(base_preds) \
                .reshape(self.num_instances * self.num_realizations, -1, len(base_models)) \
                .transpose(1, 2)
            # print(f"reshaped base_preds: {base_preds.size(), base_preds}")
            # print(f"index")
            base_preds = base_preds[torch.arange(base_preds.size(0)), index]
            # print(f"indexed base_preds: {base_preds.size(), base_preds}")

        neighbor_mask = torch.cat(
            [
                batch.neighbors for batch in data_loader
            ]
        ) \
            .reshape(self.num_instances * self.num_realizations, -1) \
            .to(device)
        choices = _masked_argmax(base_preds, neighbor_mask, dim=1)
        choices = torch.where(choices < self.n, choices, -1)
        del base_preds
        del neighbor_mask
        del index
        return choices
        

            
    def _model_assign(
        self,
        meta_model: object,
        meta_model_type: str,
        base_models: List[torch.nn.Module],
        batch_size: Optional[int] = None
    ):
        return self._gnn_model_assign(meta_model, base_models, batch_size)
        if meta_model_type == 'gnn':
            return self._gnn_model_assign(meta_model, base_models, batch_size)
        else:
            return self._heuristic_model_assign(meta_model)
            

    def update(
        self,
        t: int,
        choices: List[torch.Tensor] 
    ):
        times = defaultdict(lambda: 0)

        start = time.perf_counter()
        info_acq = time.perf_counter()
        states = [
            real_state
            for execution_state in self.execution_states
            for real_state in execution_state.state_realizations
        ]

        coin_flips = [
            real_state.coin_flips[t]
            for execution_state in self.execution_states
            for real_state in execution_state.state_realizations
        ]

        instances = [
            execution_state.instance for
            execution_state in self.execution_states
            for _ in execution_state.state_realizations
        ]
        info_acq_end = time.perf_counter()
        times['init'] += info_acq_end - info_acq

        for i, choice in enumerate(choices):
            if coin_flips[i]:
                state_start = time.perf_counter()
                state, pyg = states[i].update(instances[i], choice.item(), t)
                state_end = time.perf_counter()
                times['gnn state update'] += state_end - state_start
                
            else:
                state_start = time.perf_counter()
                state, pyg = states[i].update(instances[i], -1, t)
                state_end = time.perf_counter()
                times['state update'] += state_end - state_start

            times['state only'] += state
            times['pyg only'] += pyg
        end = time.perf_counter()
        times['total'] = end - start
        return times


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
        solve_start = time.perf_counter()
        lp_outputs = dp.call_model(inputs)
        solve_end = time.perf_counter()
        solve_time = solve_end - solve_start

        times = {"lp_solve": solve_time}
        for baseline in baselines: 
            times[baseline] = 0 


        for baseline in baselines:
            ratios[baseline] = np.zeros(
                shape=(self.num_instances, self.num_realizations)
            )
        
        for i, ex_state in enumerate(self.execution_states):
            kwargs["lp_rounding"]["x"] = lp_outputs[i]
            kwargs["naor_lp_rounding"]["x"] = lp_outputs[i]
            
            for j, real_state in enumerate(ex_state.state_realizations):
                coin_flips = real_state.coin_flips
                _, OPT = dp.offline_opt(ex_state.A, coin_flips)

                if OPT > 0:
                    ratios["learned"][i, j] = real_state.value / OPT
                    for baseline in baselines:
                        base_start = time.perf_counter()
                        _, value = BASELINES[baseline](
                            ex_state.instance,
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
    total_times = defaultdict(lambda: 0)
    start = time.perf_counter()
    gen_start = time.perf_counter()
    parallel_state = ParallelExecutionState(
        instances,
        num_realizations,
        rng
    )
    gen_end = time.perf_counter()

    # ===================================================================== #

    assign_time = 0
    
    gnn_start = time.perf_counter()
    for t in range(parallel_state.m):

        model_assign_start = time.perf_counter()
        choices = parallel_state._model_assign(
            meta_model,
            meta_model_type,
            base_models,
            batch_size
        )
        model_assign_end = time.perf_counter()
        assign_time += model_assign_end - model_assign_start
        
        times1 = parallel_state.update(t, choices)
        for key, val in times1.items():
            total_times[key] += times1[key]
    gnn_end = time.perf_counter()

    ratio_dict, times = parallel_state.compute_competitive_ratios(
        baselines,
        **kwargs
    )

   

    print(f"Generation time: {gen_end - gen_start}")
    print(f"GNN time: {gnn_end - gnn_start}")
    print(f"        Model assignment time: {assign_time}")
    print(f"        State update time: {dict(total_times)}")
    print(f"Baseline times: {times}")

    end = time.perf_counter()
    print(f"Total time: {end - start}")

    return ratio_dict, {}


def pp_output(ratios: dict) -> None:
    print('-- Competitive ratios --')
    for (name, approx_ratios) in ratios.items():
        print(f"{name}: {np.mean(approx_ratios).round(4)}")
    
