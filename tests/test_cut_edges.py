import numpy

from graphs.cut_edges import cut_edges


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
