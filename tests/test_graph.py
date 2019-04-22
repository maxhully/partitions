from graphs import Graph
from graphs.graph import random_spanning_tree, Edges, Neighbors
from scipy.sparse import dok_matrix
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

    def test_data_shape_must_match_matrix(self):
        matrix = dok_matrix((2, 2))
        matrix[0, 1] = 1
        matrix[1, 0] = 1
        data = pandas.DataFrame({"population": [100, 200, 300]})
        assert len(data.index) != matrix.shape[0]
        with pytest.raises(IndexError):
            Graph(matrix, data)

    def test_repr(self, graph):
        assert repr(graph) == "<Graph ['population', 'votes']>"

    def test_iter_iterates_nodes(self, graph):
        assert list(graph) == [0, 1, 2]


class TestEdges:
    def test_repr(self, graph):
        edges = Edges(graph.matrix)
        assert repr(edges) == "<Edges [(0, 1), (0, 2), (1, 2)]>"

    def test_contains_is_symmetric(self, graph):
        assert (0, 1) in graph.edges
        assert (1, 0) in graph.edges

    def test_does_not_contain_nodes_not_in_graph(self, graph):
        assert (8, 10) not in graph.edges


class TestNeighbors:
    def test_repr(self, graph):
        neighbors = Neighbors(graph.matrix)
        assert repr(neighbors) == "<Neighbors [3 nodes]>"


def test_random_spanning_tree():
    tree = random_spanning_tree(Graph.from_edges([(0, 1), (1, 2), (2, 3), (3, 0)]))
    print(tree.matrix)
    assert len(tree.nodes) == 4
    assert len(tree.edges) == 3
