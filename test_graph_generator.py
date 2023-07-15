import pytest
import networkx as nx
from util import _symmetrize

from graph_generator import sample_bipartite_graph


TEST_CONFIGS = [
    {
        'graph_type': 'ER',
        'p': 0.5,
        'weighted': True
    },
    {
        'graph_type': 'BA',
        'ba_param': 2,
        'weighted': True
    },
    {
        'graph_type': 'GEOM',
        'threshold': 0.2,
        'scaling': 0.5
    },
    {
        'graph_type': 'COMP'
    }
]


@pytest.mark.parametrize(
    "test_input, expected",
    [(config, True) for config in TEST_CONFIGS]
)
def test_bipartite(test_input, expected):
    A = sample_bipartite_graph(5, 4, **test_input)
    G = nx.from_numpy_array(_symmetrize(A))
    assert nx.is_bipartite(G) == expected
