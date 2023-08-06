from params import _Array
from util import powerset, diff, _neighbors
from scipy.optimize import linear_sum_assignment
import numpy as np


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
):
    m, n = A.shape
    offline_nodes = frozenset(np.arange(n))
    # First n are offline, next m online
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
    A: _Array,
    coin_flips: _Array,
    r: float
):
    m, n = A.shape
    offline_nodes = frozenset(np.arange(n))
    matching = []
    value = 0

    A = np.copy(A)

    for t in range(m):
        if coin_flips[t]:
            choice = np.argmax(A[t, :])
            if choice in offline_nodes and A[t, choice] > r:
                matching.append((t, choice))
                offline_nodes = diff(offline_nodes, choice)
                value += A[t, choice]
                A[:, choice] = 0

    return matching, value


def offline_opt(A: _Array, coin_flips: _Array):
    A = np.copy(A)
    m = A.shape[0]
    for t in range(m):
        if not coin_flips[t]:
            A[t, :] = 0

    row_ind, col_ind = linear_sum_assignment(A, maximize=True)
    value = A[row_ind, col_ind].sum()
    matching = [(row_ind[i], col_ind[i]) for i in range(len(row_ind))]
    return matching, value
