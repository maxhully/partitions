from graphs import Graph
from graphs.graph import random_spanning_tree
import pandas
import pytest


@pytest.fixture
def graph():
    return Graph.from_edges(
        [(0, 1), (1, 2), (0, 2)],
        data=pandas.DataFrame(
            {"population": [100, 200, 50], "votes": [50, 60, 40]}, index=[0, 1, 2]
        ),
    )


class TestGraph:
    def test_init(self, graph):
        assert graph

    def test_neighbors(self, graph):
        assert set(graph.neighbors[0]) == {1, 2}


def test_random_spanning_tree():
    tree = random_spanning_tree(Graph.from_edges([(0, 1), (1, 2), (2, 3), (3, 0)]))
    print(tree.matrix)
    assert len(tree.nodes) == 4
    assert len(tree.edges) == 3
