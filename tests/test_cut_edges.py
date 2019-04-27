from collections import Counter

import numpy

from graphs.cut_edges import cut_edges, cut_edges_for_subset, Boundary


def check_cut_edges(graph, assignment, expected):
    result = cut_edges(graph.edges, numpy.asarray(assignment))
    assert set(result) == expected


class TestCutEdges:
    def test_four_cycle(self, four_cycle):
        check_cut_edges(four_cycle, [0, 0, 1, 1], {(0, 3), (1, 2)})

    def test_nonregular(self, nonregular):
        check_cut_edges(
            nonregular, [0, 0, 1, 2, 1, 2], {(1, 3), (1, 5), (1, 2), (4, 5), (2, 5)}
        )

    def test_check_cut_edges_for_subset(self, nonregular):
        subset = numpy.asarray([0, 1, 3])
        expected_edges = {(1, 2), (1, 5), (3, 5)}

        result = cut_edges_for_subset(nonregular, subset)
        assert set(result) == expected_edges

    def test_cut_edges_for_decreasing_edges(self, four_cycle):
        result = cut_edges_for_subset(four_cycle, numpy.asarray([2, 3]))
        assert set(result) == {(0, 3), (1, 2)}

    def test_cut_edges_for_decreasing_edges_nonregular(self, nonregular):
        result = cut_edges_for_subset(nonregular, numpy.asarray([0, 2, 5]))
        assert set(result) == {(0, 1), (1, 2), (2, 4), (3, 5), (4, 5), (1, 5)}


class TestBoundary:
    def test_boundary_neighbors(self, four_cycle):
        boundary = Boundary(four_cycle, numpy.array([2, 3]))
        assert set(boundary.neighbors) == {0, 1}

    def test_boundary_nodes(self, nonregular):
        boundary = Boundary(nonregular, numpy.asarray([4, 2, 5]))
        assert set(boundary.nodes) == {2, 5}

    def test_unique_boundary_nodes(self, nonregular):
        boundary = Boundary(nonregular, numpy.asarray([4, 2, 5]))
        counts = Counter(boundary.nodes)
        assert counts == {2: 1, 5: 2}
        assert list(boundary.unique_nodes()) == [2, 5]

    def test_unique_boundary_neighbors(self, nonregular):
        boundary = Boundary(nonregular, numpy.asarray([4, 2, 5]))
        counts = Counter(boundary.neighbors)
        assert counts == {1: 2, 3: 1}
        assert list(boundary.unique_neighbors()) == [1, 3]
