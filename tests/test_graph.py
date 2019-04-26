from graphs import Graph
from graphs.graph import Edges, Neighbors, subgraph_matrix
from graphs.tree import random_spanning_tree
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


@pytest.fixture
def k4():
    return Graph.from_edges([(0, 1), (1, 2), (2, 3), (3, 0), (0, 2), (1, 3)])


@pytest.fixture
def nonregular():
    """
    0  3
    | /|
    1--5
    | /|
    2--4
    """
    return Graph.from_edges(
        [(0, 1), (1, 2), (5, 1), (1, 3), (4, 2), (5, 2), (5, 4), (3, 5)]
    )


@pytest.fixture
def four_cycle():
    return Graph.from_edges([(0, 1), (1, 2), (2, 3), (3, 0)])


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

    def test_len_equals_number_of_nodes(self, nonregular):
        assert len(nonregular) == 6

    def test_can_create_from_an_iterable_of_edges(self):
        graph = Graph.from_edges((i, i + 1) for i in range(10))
        assert len(graph) == 11


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

    def test_neighbors_k4(self, k4):
        neighbors = Neighbors(k4.matrix)
        assert set(neighbors[0]) == {1, 2, 3}
        assert set(neighbors[1]) == {0, 2, 3}
        assert set(neighbors[2]) == {0, 1, 3}
        assert set(neighbors[3]) == {1, 2, 0}
        for node in range(4):
            assert len(neighbors[node]) == 3

    def test_neighbors_nonregular(self, nonregular):
        neighbors = Neighbors(nonregular.matrix)
        assert set(neighbors[0]) == {1}
        assert set(neighbors[1]) == {0, 3, 5, 2}
        assert set(neighbors[2]) == {1, 4, 5}
        assert set(neighbors[3]) == {1, 5}
        assert set(neighbors[4]) == {2, 5}
        assert set(neighbors[5]) == {1, 2, 3, 4}


class TestRandomSpanningTree:
    def test_on_four_cycle(self, four_cycle):
        tree = random_spanning_tree(four_cycle)
        assert len(tree.nodes) == 4
        assert len(tree.edges) == 3

    def test_on_nonregular(self, nonregular):
        tree = random_spanning_tree(nonregular)
        assert len(tree.nodes) == 6
        assert len(tree.edges) == 5
        # This edge has to be in it, because 0 is a leaf
        assert (0, 1) in tree.edges
        assert (1, 0) in tree.edges
        # One of these must be in it
        assert (1, 3) in tree.edges or (3, 5) in tree.edges
        # One of these must be in it
        assert any(edge in tree.edges for edge in [(2, 4), (2, 5), (2, 1)])

        for node in nonregular:
            assert any(
                (node, neighbor) in tree.edges
                for neighbor in nonregular.neighbors[node]
            )


def test_subgraph_matrix(four_cycle):
    nodes = [1, 2, 3]
    matrix = subgraph_matrix(four_cycle.matrix, nodes)
    print(matrix.toarray())
    assert matrix.shape == (3, 3)
