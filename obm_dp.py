from itertools import chain, combinations
from params import _Array
from typing import Optional
import numpy as np


def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    powerset = chain.from_iterable(combinations(s, r) for r in range(len(s)+1))
    return list(map(set, powerset))


def diff(S: frozenset, u: int) -> frozenset:
    return S.difference(set([u]))


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
        for u in S:
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

        for u in S:
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

    n = A.shape[1]
    hint = np.zeros(n + 1)
    hint[-1] = cache[t + 1][offline_nodes][0]
    for u in offline_nodes:
        hint[u] = cache[t + 1][diff(offline_nodes, u)][0] + A[t, u]

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
            hint = one_step_stochastic_opt(A, offline_nodes, t, cache)
            choice = np.argmax(hint)
            if choice in offline_nodes:
                matching.append((t, choice))
                offline_nodes = diff(offline_nodes, choice)
                value += A[t, choice]

    return matching, value


def greedy(
    A: _Array,
    coin_flips: _Array
):
    m, n = A.shape
    offline_nodes = frozenset(np.arange(n))
    matching = []
    value = 0

    def _mask(offline_nodes, adj):
        return [adj[i] if i in offline_nodes else 0 for i in range(n)]

    for t in range(m):
        if coin_flips[t]:
            masked_adj = _mask(offline_nodes, A[t, :])
            choice = np.argmax(masked_adj)
            if choice in offline_nodes:
                matching.append((t, choice))
                offline_nodes = diff(offline_nodes, choice)
                value += A[t, choice]

    return matching, value
