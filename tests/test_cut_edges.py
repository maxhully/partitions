import numpy

from graphs.cut_edges import cut_edges, cut_edges_for_subset


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

        assignment = numpy.asarray([0, 0, 1, 0, 2, 2])
        result = cut_edges_for_subset(nonregular, subset, assignment)
        assert set(result) == expected_edges
