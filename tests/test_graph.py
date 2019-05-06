import numpy
import pandas
import pytest
from scipy.sparse import csr_matrix, dok_matrix

from partitions import Graph
from partitions.graph import Edges, EmbeddedGraph, Neighbors, subgraph_matrix


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
            Graph.from_matrix(matrix, data)

    def test_repr(self, graph):
        assert repr(graph) == "<Graph ['population', 'votes']>"

    def test_iter_iterates_nodes(self, graph):
        assert list(graph) == [0, 1, 2]

    def test_len_equals_number_of_nodes(self, nonregular):
        assert len(nonregular) == 6

    def test_can_create_from_an_iterable_of_edges(self):
        graph = Graph.from_edges((i, i + 1) for i in range(10))
        assert len(graph) == 11

    def test_from_matrix_casts_matrix_to_bool(self, matrix):
        graph = Graph.from_matrix(matrix)
        assert graph.matrix.dtype == bool

    def test_constructor_keeps_original_matrix(self, matrix):
        graph = Graph(matrix)
        assert graph.matrix is matrix

    def test_subgraph(self, nonregular):
        subgraph = nonregular.subgraph({0, 1, 5, 2})
        assert isinstance(subgraph, EmbeddedGraph)
        assert set(subgraph.image[subgraph.nodes]) == {0, 1, 2, 5}

        embedded_edges = {tuple(subgraph.image[list(edge)]) for edge in subgraph.edges}
        assert embedded_edges == {(1, 5), (0, 1), (2, 5), (1, 2)}


class TestEdges:
    def test_repr(self, graph):
        edges = Edges(graph.matrix)
        assert repr(edges) == "<Edges [(0, 1), (0, 2), (1, 2)]>"

    def test_contains_is_symmetric(self, graph):
        assert (0, 1) in graph.edges
        assert (1, 0) in graph.edges

    def test_does_not_contain_nodes_not_in_graph(self, graph):
        assert (8, 10) not in graph.edges

    def test_stores_matrix_as_csr(self, matrix):
        edges = Edges(matrix)
        assert isinstance(edges.matrix, csr_matrix)

    def test_matrix_is_upper_triangular(self):
        matrix = numpy.array([[0, 1, 1], [0, 0, 1], [0, 0, 0]])
        expected_support = matrix > 0

        edges = Edges(matrix)
        actual_support = edges.matrix.toarray() > 0
        assert numpy.all(actual_support == expected_support)

        symmetric_matrix = numpy.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
        edges = Edges(symmetric_matrix)
        actual_support = edges.matrix.toarray() > 0
        assert numpy.all(actual_support == expected_support)


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

    def test_stores_matrix_as_csr(self, matrix):
        neighbors = Neighbors(matrix)
        assert isinstance(neighbors.matrix, csr_matrix)

    def test_implements_sequence(self, matrix):
        neighbors = Neighbors(matrix)
        # __len__
        assert len(neighbors) == 3
        # __getitem__
        assert all(neighbors[i] is not None for i in range(len(neighbors)))
        # __iter__
        assert list(neighbors)

    def test_degrees(self, k4, nonregular, four_cycle):
        neighbors = Neighbors(k4.matrix)
        assert numpy.all(neighbors.degrees() == numpy.asarray([3, 3, 3, 3]))

        neighbors = Neighbors(four_cycle.matrix)
        assert numpy.all(neighbors.degrees() == numpy.asarray([2, 2, 2, 2]))

        neighbors = Neighbors(nonregular.matrix)
        assert numpy.all(neighbors.degrees() == numpy.asarray([1, 4, 3, 2, 2, 4]))

    def test_can_have_edge_data(self):
        edge_index = pandas.MultiIndex.from_tuples([(0, 1), (1, 2), (2, 3)])
        edges = pandas.DataFrame({"length": [10, 20, 30]}, index=edge_index)
        graph = Graph.from_edges(edges)
        assert set(graph.edges.data.index) == set(graph.edges)

    def test_raises_if_edge_data_indexed_incorrectly(self):
        edge_index = pandas.MultiIndex.from_tuples([(1, 0), (1, 2), (3, 2)])
        edges = pandas.DataFrame({"length": [10, 20, 30]}, index=edge_index)
        with pytest.raises(ValueError):
            Graph.from_edges(edges)


class TestSubgraphMatrix:
    def test_matrix_shape(self, four_cycle):
        nodes = [1, 2, 3]
        matrix = subgraph_matrix(four_cycle.matrix, nodes)
        assert matrix.shape == (3, 3)

    def test_adjacency_structure(self, nonregular):
        nodes = [0, 1, 3]
        matrix = subgraph_matrix(nonregular.matrix, nodes)

        expected = dok_matrix((3, 3))
        expected[0, 1] = 1
        expected[1, 0] = 1
        expected[1, 2] = 1
        expected[2, 1] = 1

        assert numpy.all(matrix.toarray() == expected.toarray())


class TestEmbeddedGraph:
    def test_repr(self, k4):
        subgraph = k4.subgraph([1, 2, 3])
        assert repr(subgraph) == "<EmbeddedGraph [3 nodes]>"

    def test_aggregates_boundary_data_if_graph_has_edge_data(self, edges_with_data):
        graph = Graph.from_edges(edges_with_data)
        subgraph = graph.subgraph([0, 1])
        expected = edges_with_data["length"][0, 3] + edges_with_data["length"][1, 2]
        assert set(subgraph.cut_edges) == {(0, 3), (1, 2)}
        assert subgraph.boundary.data["length"] == expected

    def test_subgraph_of_subgraph_is_a_subgraph_of_the_original_graph(self, k4):
        subgraph = k4.subgraph([1, 2, 3])
        subgraph_of_subgraph = subgraph.subgraph([0, 1])
        assert set(subgraph_of_subgraph.image) == {1, 2}
        assert subgraph_of_subgraph.graph is k4

    def test_disjoin_union(self, nonregular):
        subgraph1 = nonregular.subgraph([0, 1, 3])
        subgraph2 = nonregular.subgraph([2, 5])
        union = subgraph1.union(subgraph2, disjoint=True)
        image_of_edges = [(union.image[i], union.image[j]) for i, j in union.edges]
        assert len(union.edges) == 6
        assert set(image_of_edges) == {(0, 1), (1, 2), (3, 5), (1, 3), (1, 5), (2, 5)}
        assert len(union.image) == 5
        assert set(union.image) == {0, 1, 2, 3, 5}

    def test_union(self, nonregular):
        subgraph1 = nonregular.subgraph([0, 1, 2, 3])
        subgraph2 = nonregular.subgraph([1, 2, 5])
        union = subgraph1.union(subgraph2)
        image_of_edges = [(union.image[i], union.image[j]) for i, j in union.edges]
        assert len(union.edges) == 6
        assert set(image_of_edges) == {(0, 1), (1, 2), (3, 5), (1, 3), (1, 5), (2, 5)}
        assert len(union.image) == 5
        assert set(union.image) == {0, 1, 2, 3, 5}

    def test_unions_only_work_if_from_same_graph(self, k4, nonregular):
        subgraph1 = k4.subgraph([2])
        subgraph2 = nonregular.subgraph([3])
        with pytest.raises(ValueError):
            subgraph1.union(subgraph2)
        with pytest.raises(ValueError):
            subgraph1.union(subgraph2, disjoint=True)
