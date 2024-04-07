import gurobipy as gp
import itertools as it
import multiprocessing as mp
import numpy as np

from numpy.random import Generator
from scipy.optimize import linear_sum_assignment
from typing import Optional, Tuple

from params import _Array, _Instance, _Matching
from util import powerset, diff, _neighbors



def cache_stochastic_opt(A: _Array, p: _Array) -> dict:
    '''
    Computes the value-to-go for all timesteps t=1...m and for
    all subsets S of offline nodes {1, ..., n}, according to
    online arrival probabilities [p] and underlying graph
    adjacency matrix [A].
    '''

    def _neighbor_max_argmax(S: frozenset, t: int):
        argmax = -1
        max_val = cache[t + 1][S][0]
        for u in _neighbors(A, S, t):
            val = cache[t + 1][diff(S, u)][0] + A[t, u]
            if val > max_val:
                argmax = u
                max_val = val

        return max_val, argmax

    def _value_to_go(S: frozenset, t: int):
        '''
        Computes the value-to-go of unmatched node set [S] starting
        at online node [t] for the graph defined by the adjacency matrix
        [A], caching all intermediate results.
        '''
        if S not in cache[t + 1]:
            cache[t + 1][S] = _value_to_go(S, t + 1)

        for u in _neighbors(A, S, t):
            S_diff_u = diff(S, u)
            if S_diff_u not in cache[t + 1]:
                cache[t + 1][S_diff_u] = _value_to_go(S_diff_u, t + 1)

        max_val, argmax = _neighbor_max_argmax(S, t)

        exp_value_to_go = (1 - p[t]) * cache[t + 1][S][0] + \
            p[t] * max([cache[t + 1][S][0], max_val])

        return (exp_value_to_go, argmax)

    m, n = A.shape
    offline_nodes = frozenset(np.arange(n))
    cache = {t: dict() for t in np.arange(m + 1)}

    # Set boundary conditions
    for t in np.arange(m + 1):
        cache[t][frozenset()] = (0, None)
    for subset in powerset(offline_nodes):
        cache[m][frozenset(subset)] = (0, None)

    # Cache all relevant DP quantities
    cache[0][frozenset(offline_nodes)] = _value_to_go(offline_nodes, 0)
    return cache


def one_step_stochastic_opt(
    A: _Array,
    offline_nodes: frozenset,
    t: int,
    cache: dict
) -> _Array:
    '''For the graph defined by adjacency matrix [A], computes the value-to-go
    associated with each offline node matching to online node [t]. Nodes which
    are not active in [offline_nodes] have a value-to-go of 0.'''

    N_t = _neighbors(A, offline_nodes, t)
    hint = np.array([
        *[
            cache[t + 1][diff(offline_nodes, u)][0] + A[t, u]
            for u in N_t
        ],
        cache[t + 1][offline_nodes][0]
    ])

    return hint


def stochastic_opt(
    A: _Array,
    coin_flips: _Array,
    cache: dict
) -> Tuple[_Matching, float]:
    
    m, n = A.shape
    offline_nodes = frozenset(np.arange(n))
    matching = []
    value = 0

    for t in range(m):
        if coin_flips[t]:
            choice = cache[t][offline_nodes][1]
            if choice in offline_nodes:
                matching.append((t, choice))
                offline_nodes = diff(offline_nodes, choice)
                value += A[t, choice]

    return matching, value

    
def greedy(
    instance: _Instance,
    coin_flips: _Array
) -> Tuple[_Matching, float]:
    
    return threshold_greedy(instance, coin_flips, 0)

def threshold_greedy(
    instance: _Instance,
    coin_flips: _Array,
    threshold: float
) -> Tuple[_Matching, float]:
    A, _, noisy_A, _ = instance
    m, n = A.shape

    offline_nodes = frozenset(np.arange(n))
    matching = []
    value = 0

    noisy_A = np.copy(noisy_A)

    for t in range(m):
        if coin_flips[t]:
            choice = np.argmax(noisy_A[t, :])
            if choice in offline_nodes and noisy_A[t, choice] > threshold:
                matching.append((t, choice))
                offline_nodes = diff(offline_nodes, choice)
                value += A[t, choice]
                noisy_A[:, choice] = 0

    return matching, value


def offline_opt(A: _Array, coin_flips: _Array) -> Tuple[_Matching, float]:
    A = np.copy(A)
    m = A.shape[0]
    for t in range(m):
        if not coin_flips[t]:
            A[t, :] = 0

    row_ind, col_ind = linear_sum_assignment(A, maximize=True)
    value = A[row_ind, col_ind].sum()
    matching = [(row_ind[i], col_ind[i]) for i in range(len(row_ind))]
    return matching, value


def _build_variables(model: gp.Model, indices: list) -> gp.Var:
    return model.addVars(
        indices,
        vtype=gp.GRB.CONTINUOUS,
        lb=0,
        name='x'
    )

def _build_constraints(
    model: gp.Model,
    x: gp.Var,
    p: _Array,
    online_nodes: list,
    offline_nodes: list,
    indices: list
) -> None:
    
    model.addConstrs(
        (
            gp.quicksum(x[(i, t)] for t in online_nodes)
            <= 1
            for i in offline_nodes
        ),
        name="offline_matching"
    )
    
    model.addConstrs(
        (
            gp.quicksum(x[(i, t)] for i in offline_nodes)
            <= p[t]
            for t in online_nodes
        ),
        name="online_matching"
    )
    
    model.addConstrs(
        (
            x[(i, t)] - 
            p[t] * (1 - gp.quicksum(x[i, s] for s in online_nodes[:t])) 
            <= 0
            for (i, t) in indices
        ),
        name="online_knowledge"
    )

def _build_objective(
    model: gp.Model,
    x: gp.Var,
    A: _Array,
    indices: list
) -> None:
    
    model.setObjective(
        gp.quicksum(x[(i, t)] * A[t, i] for (i, t) in indices),
        gp.GRB.MAXIMIZE
    )


def _lp_match(input, verbose: Optional[bool] = False) -> _Array:

    noisy_A, noisy_p = input
    m, n = noisy_A.shape
    online_nodes = range(m)
    offline_nodes = range(n)
    indices = list(it.product(offline_nodes, online_nodes))

    model = gp.Model('LP-MATCH', env=gp.Env())
    x = _build_variables(model, indices)
    _build_constraints(model, x, noisy_p, online_nodes, offline_nodes, indices)
    _build_objective(model, x, noisy_A, indices)
   
    if not verbose:
        model.Params.LogToConsole = 0
    model.optimize()

    output = np.array([x[(i, t)].x for (i, t) in indices]).reshape(n, m).T
    return output


def parallel_solve(inputs):
    pool = mp.Pool(mp.cpu_count())
    return np.array(pool.map(_lp_match, inputs))


def _scale_mat_by_vec(A: _Array, b: _Array) -> _Array:
    ''' 
    Given a matrix [A] and vector [b], gives the matrix formed by multiplying
    the first row of A by b_1, the second row of A by b_2, and so on.
    '''
    return (A.T * b).T


def _compute_s(x: _Array):
    return np.cumsum(
        np.vstack([np.zeros(shape=(1, x.shape[1])), x]),
        axis=0
    )[:-1]


def _compute_proposal_probs(x, p):
    eps = np.finfo(np.float32).eps
    
    s = _compute_s(x)
    denom = _scale_mat_by_vec(1 - s, p)
    proposal_probs = x / denom
    
    proposal_probs = np.where(denom > eps, proposal_probs, 0)
    proposal_probs = np.where(proposal_probs <= 1, proposal_probs, 1)
    proposal_probs = np.round(proposal_probs, 5)
    proposal_probs = np.where(proposal_probs >= 0, proposal_probs, 0)
    
    assert (np.all(proposal_probs >= 0)), f"{proposal_probs[proposal_probs < 0]}"
    assert (np.all(proposal_probs <= 1)), f"{proposal_probs[proposal_probs > 1]}"
    
    return proposal_probs


def _naor_scaling(x, s, eps, delta, theta):
    a = np.maximum(theta, s)
    b = np.maximum(theta, s + x)
    return x * (1 - eps) + (b - a) * (eps + delta)

def _first_fit(values, mask):
    m, n = values.shape
    bins = {t: [[]] for t in range(m)}
    singleton_bins = {t: [] for t in range(m)}
    current_bin_val = {t: 0 for t in range(m)}

    for t in range(m):
        for i in range(n):
            value = values[t, i]
            if mask[t, i]:
                if value + current_bin_val[t] > 1:
                    bins[t].append([(i, value)])
                    current_bin_val[t] = value
                else:
                    bins[t][-1].append((i, value))
                    current_bin_val[t] += value
            else:
                singleton_bins[t].append([(i, value)])

    all_bins = {
        t: [*singleton_bins[t], *bins[t]]
        for t in bins.keys()
    }
    return all_bins


def _get_candidates(m, n, bins, rng):
    candidates = np.zeros(shape=(m, n))
    for t in bins.keys():
        u = rng.uniform(0, 1)
        for bin in bins[t]:
            cum_value = 0
            for i, value in bin:
                cum_value += value
                if cum_value > u:
                    candidates[t, i] = 1
    return candidates.astype(bool)

def _lp_approx(
    x: _Array,
    instance: _Instance,
    coin_flips: _Array,
    proposal_prob_fn: callable,
    candidate_fn: callable,
    rng: Generator
) -> Tuple[_Matching, float]:
    
    A, _, noisy_A, noisy_p = instance
    m, n = x.shape

    matching = []
    val = 0
    avail_mask = np.array(n * [True])

    proposal_probs = proposal_prob_fn(x, noisy_p)
    candidates = candidate_fn(proposal_probs, rng)

    for t in range(m):
        if coin_flips[t]:
            proposals = candidates[t, :]
            valid_proposals = np.bitwise_and(avail_mask, proposals)

            if not np.all(valid_proposals == 0):
                matched_node = np.argmax(np.multiply(noisy_A[t], valid_proposals))
                matching.append((t, matched_node))
                val += A[t, matched_node]
                avail_mask[matched_node] = 0
    return matching, val


def braverman_lp_approx(
    x: _Array,
    instance: _Instance,
    coin_flips: _Array,
    rng: Generator
) -> Tuple[_Matching, float]:
    
    def _candidate_fn(proposal_probs, rng):
        return rng.binomial(1, proposal_probs)
    
    def _proposal_prob_fn(x, p):
        return _compute_proposal_probs(x, p)
    
    return _lp_approx(
        x,
        instance,
        coin_flips,
        _proposal_prob_fn,
        _candidate_fn,
        rng
    )


def naor_lp_approx(
    x: _Array,
    instance: _Instance,
    coin_flips: _Array,
    rng: Generator
) -> Tuple[_Matching, float]:

    def _proposal_prob_fn(x, p):
        eps = 0.0480
        delta = 0.0643
        theta = delta / (eps + delta)

        s = _compute_s(x)
        x_hat = _naor_scaling(x, s, eps, delta, theta)
        return _compute_proposal_probs(x_hat, p)
    
    def _candidate_fn(proposal_probs, rng):
        eps = 0.0480
        delta = 0.0643
        theta = delta / (eps + delta)
        m, n = proposal_probs.shape

        bins = _first_fit(proposal_probs, (proposal_probs <= theta))
        return _get_candidates(m, n, bins, rng)
    
    return _lp_approx(
        x,
        instance,
        coin_flips,
        _proposal_prob_fn,
        _candidate_fn,
        rng
    )
