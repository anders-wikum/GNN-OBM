from itertools import chain, combinations
import numpy as np
from params import _Array


def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    powerset = chain.from_iterable(combinations(s, r) for r in range(len(s)+1))
    return list(map(set, powerset))


def diff(S: frozenset, u: int) -> frozenset:
    return S.difference(set([u]))


def _random_subset(seq, m):
    """Return m unique elements from seq.

    This differs from random.sample which can return repeated
    elements if seq holds repeated elements.
    """
    targets = set()
    while len(targets) < m:
        x = np.random.choice(seq)
        targets.add(x)

    return targets


def _symmetrize(adj):
    n, m = adj.shape
    A = np.zeros((n + m, n + m))
    A[:n, n:] = adj
    A[n:, :n] = adj.T
    return A


def _extract_edges(A: _Array):
    m, n = A.shape
    edge_index = []
    edge_attr = []

    # Edges/edge weights in underlying graph
    for i in range(m):
        for j in range(n):
            if A[i, j] > 0:
                edge_index.append([i, j])
                edge_attr.append([A[i, j]])

    # Virtual node representing no match
    for j in range(n):
        edge_index.append([n + m, j])
        edge_attr.append([0])

    return edge_index, edge_attr


def _neighbors(A: _Array, S: set, t: int) -> _Array:
    return [u for u in S if A[t, u] > 0]
