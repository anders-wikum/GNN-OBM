import traceback
import numpy as np
from scipy.special import rel_entr
from params import _Array
from util import diff, _neighbors


def entropy(p):
    return -np.sum(p * np.log2(p, where=(p != 0)))


def uniform_entropy(size):
    if size <= 1:
        return 1
    else:
        p = np.array([1 / size for _ in range(size)])
        return entropy(p)


def hints_of_node(S: frozenset, t: int, cache: dict, A: _Array):
    try:
        N_t = _neighbors(A, S, t)
        p = np.array([
            *[
                cache[t + 1][diff(S, u)][0] + A[t, u]
                if u in N_t
                else -1
                for u in S
            ],
            cache[t + 1][S][0]
        ])
        p = np.array(p)

        if np.all(p == 0):
            p = p + 1
    except:
        # Should never be triggered. We are ignoring the last layer.
        traceback.print_exc()
        p = np.array([1])
    return p


def rel_entropy_of_node(S: frozenset, t: int, cache: dict, A: _Array):
    p = hints_of_node(S, t, cache, A)
    p = p[p >= 0]
    if sum(p) == 0:
        return 0
    p = p / sum(p)
    num_choices = len(p)
    uniform_dist = [1 / num_choices for _ in range(num_choices)]
    return sum(rel_entr(p, uniform_dist))


def greedy_suboptimality_of_node(S: frozenset, t: int, cache: dict, A: _Array):
    n = A.shape[1]
    values_to_go = hints_of_node(S, t, cache, A)
    edge_weights = np.array([A[t, u] for u in S])
    opt_vtg = np.max(values_to_go)
    greedy_decision = np.argmax(edge_weights)
    return (opt_vtg - values_to_go[greedy_decision]) / \
        cache[0][frozenset(np.arange(n))][0]


def uniform_suboptimality_of_node(S: frozenset, t: int, cache: dict, A: _Array):
    n = A.shape[1]
    values_to_go = hints_of_node(S, t, cache, A)
    opt_vtg = np.max(values_to_go)
    uniform_decision = np.random.choice(np.arange(len(values_to_go)))
    return (opt_vtg - values_to_go[uniform_decision]) / \
        cache[0][frozenset(np.arange(n))][0]


def greedy_correctness_of_node(S: frozenset, t: int, cache: dict, A: _Array):
    values_to_go = hints_of_node(S, t, cache, A)
    edge_weights = np.array([A[t, u] for u in S])
    opt_decision = np.argmax(values_to_go)
    greedy_decision = np.argmax(edge_weights)
    return 1 * (greedy_decision == opt_decision)


def quantity_of_cache_layer(cache: dict, t: int, node_func: callable, A: _Array):
    return np.array([node_func(S, t, cache, A) for S in cache[t].keys() if len(S) > 0])


def expected_quantities(cache: dict, node_func: callable, A: _Array):
    layers = list(cache.keys())[:-1]
    return layers, [
        np.nanmean(quantity_of_cache_layer(cache, t, node_func, A)[1:])
        for t in layers
    ]


NODE_FUNCS = {
    'relative_entropy': rel_entropy_of_node,
    'greedy_suboptimality': greedy_suboptimality_of_node,
    'greedy_correctness': greedy_correctness_of_node,
    'uniform_suboptimality': uniform_suboptimality_of_node
}
