from itertools import chain, combinations


def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    powerset = chain.from_iterable(combinations(s, r) for r in range(len(s)+1))
    return list(map(set, powerset))


def diff(S: frozenset, u: int) -> frozenset:
    return S.difference(set([u]))
