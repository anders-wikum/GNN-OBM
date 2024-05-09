import time
import torch
import numpy as np

from collections import defaultdict
from numpy.random import Generator
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch
from typing import Optional, List, Tuple

from algorithms import greedy, threshold_greedy, braverman_lp_approx, \
    naor_lp_approx, pollner_lp_approx, offline_opt, parallel_solve
from torch_converter import init_pyg, update_pyg
from params import _Instance
from util import Dataset, diff, _flip_coins, _masked_argmax


BASELINES = {
    'greedy': greedy,
    'greedy_t': threshold_greedy,
    'lp_rounding': braverman_lp_approx,
    'naor_lp_rounding': naor_lp_approx,
    'pollner_lp_rounding': pollner_lp_approx
}


class StateRealization:
    def __init__(self, instance: _Instance, rng: Generator):
        (A, p, _, _) = instance
        self.value = 0
        self.matching = []
        self.offline_nodes = frozenset(np.arange(A.shape[1]))
        self.dataset = init_pyg(instance, rng)
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
            update_pyg(
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
        self.A, self.p, self.noisy_A, self.noisy_p = instance
        self.instance = instance
  
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


    def _get_default_model_index(self):
        return np.zeros(
            self.num_instances * self.num_realizations
        ).astype(int)
    

    def _get_threshold_model_index(self):
        online_offline_ratios = np.array([
            realized_state.dataset.graph_features[1].item()
            for execution_state in self.execution_states
            for realized_state in execution_state.state_realizations
        ])
        return (online_offline_ratios > 1.5).astype(int)


    def _get_base_predictions(
        self,
        batch: Batch,
        base_models: List[torch.nn.Module]
    ):
        
        with torch.no_grad():
            base_pred = torch.concat(
                [
                    base_model(batch)
                    for base_model in base_models
                ],
                dim=1
            )
            return base_pred
    

    def _get_predictions(
        self, 
        data_loader: DataLoader,
        meta_model: torch.nn.Module,
        base_models: List[torch.nn.Module]
    ):
        if meta_model is None:
            index = self._get_default_model_index()

        device = next(base_models[0].parameters()).device
        with torch.no_grad():
            base_preds = []
            model_preds = []

            for batch in data_loader:
                batch.to(device)

                base_pred = self._get_base_predictions(batch, base_models)
                base_preds.append(base_pred)
                batch.base_model_preds = base_pred

                if meta_model is not None:
                    model_pred = torch.argmax(meta_model(batch), dim=1).int()
                    model_preds.append(model_pred)

        return torch.cat(base_preds), torch.cat(model_preds)

    @staticmethod
    def _select_predictions_by_index(
        base_preds: torch.Tensor, 
        index: torch.Tensor,
        num_graphs: int,
        num_models: int
    ) -> torch.Tensor:
        
        base_preds = base_preds \
            .reshape(num_graphs, -1, num_models) \
            .transpose(1, 2)
    
        return base_preds[torch.arange(base_preds.size(0)), index]
    

    def _build_loader(self, batch_size: int) -> Tuple[DataLoader, int]:

        data = Dataset([
            realized_state.dataset
            for execution_state in self.execution_states
            for realized_state in execution_state.state_realizations
        ])

        loader =  DataLoader(
            data,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0
        )
        return loader, len(data)

        
    def compute_choices(
        self,
        meta_model: torch.nn.Module,
        base_models: List[torch.nn.Module],
        batch_size: int
    ):

        data_loader, num_graphs = self._build_loader(batch_size)

        base_preds, index = self._get_predictions(
            data_loader=data_loader,
            meta_model=meta_model,
            base_models=base_models
        )
        
        preds = self._select_predictions_by_index(
            base_preds,
            index,
            num_graphs,
            len(base_models)
        ).to('cpu')
       
        del index

        neighbor_mask = torch.cat([batch.neighbors for batch in data_loader]) \
            .reshape(num_graphs, -1)
        choices = _masked_argmax(preds, neighbor_mask, dim=1)
        choices = torch.where(choices < self.n, choices, -1)

        del preds
        del base_preds
        del neighbor_mask

        return choices
            
    def _get_update_info(self, t: int) -> Tuple[list, list, list]:
        states = []
        coin_flips = []
        instances = []
        for execution_state in self.execution_states:
            for real_state in execution_state.state_realizations:
                states.append(real_state)
                coin_flips.append(real_state.coin_flips[t])
                instances.append(execution_state.instance)

        return states, coin_flips, instances

    def update(
        self,
        t: int,
        choices: List[torch.Tensor] 
    ):
        times = defaultdict(lambda: 0)

        start = time.perf_counter()
        info_acq = time.perf_counter()
        states , coin_flips, instances = self._get_update_info(t)

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


    def compute_competitive_ratios(
        self,
        baselines: List[str],
        baselines_only: bool,
        **kwargs
    ):
    
        ratios = {
            "learned": np.zeros(
                shape=(self.num_instances, self.num_realizations)
            )
        }
        if len(baselines) > 0:
            inputs = [
                (instance[2], instance[3])
                for instance in self.instances
            ]
            solve_start = time.perf_counter()
            lp_outputs = parallel_solve(inputs)
            solve_end = time.perf_counter()
            solve_time = solve_end - solve_start
        else:
            solve_time = 0
            lp_outputs = [None] * self.num_instances

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
            kwargs["pollner_lp_rounding"]["x"] = lp_outputs[i]
            
            for j, real_state in enumerate(ex_state.state_realizations):
                _, OPT = offline_opt(ex_state.A, real_state.coin_flips)

                if OPT > 0:
                    ratios["learned"][i, j] = real_state.value / OPT
                    for baseline in baselines:
                        base_start = time.perf_counter()
                        _, value = BASELINES[baseline](
                            instance=ex_state.instance,
                            coin_flips=real_state.coin_flips,
                            **kwargs[baseline]
                        )
                        ratios[baseline][i, j] = value / OPT
                        base_end = time.perf_counter()
                        times[baseline] += base_end - base_start
                        
                else:
                    for method in ratios.keys():
                        ratios[method][i, j] = np.nan

        if baselines_only:         
            del ratios["learned"]
        avg_ratios = {
            name: list(np.nanmean(ratio_matrix, axis=1))
            for (name, ratio_matrix) in ratios.items()
        }

        return avg_ratios, times


def evaluate_model(
    meta_model: object,
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

    baselines_only = (len(base_models) == 0)
    assign_time = 0
    
    gnn_start = time.perf_counter()

    if not baselines_only:
        for t in range(parallel_state.m):

            model_assign_start = time.perf_counter()
            choices = parallel_state.compute_choices(
                meta_model,
                base_models,
                batch_size
            )
            model_assign_end = time.perf_counter()
            assign_time += model_assign_end - model_assign_start
            
            times1 = parallel_state.update(t, choices)
            for key, _ in times1.items():
                total_times[key] += times1[key]
    gnn_end = time.perf_counter()

    ratio_dict, times = parallel_state.compute_competitive_ratios(
        baselines,
        baselines_only,
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
    
