from itertools import combinations

import pandas
import pytest
from scipy.sparse import dok_matrix

from partitions import Graph, Partition


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
def k8():
    return Graph.from_edges(combinations(range(8), 2))


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


@pytest.fixture
def matrix():
    matrix = dok_matrix((3, 3))
    matrix[0, 1] = 1
    matrix[1, 2] = 1
    return matrix


@pytest.fixture
def edges_with_data():
    edge_index = pandas.MultiIndex.from_tuples([(0, 1), (1, 2), (2, 3), (0, 3)])
    edges = pandas.DataFrame({"length": [10, 21, 33, 44]}, index=edge_index)
    return edges


@pytest.fixture
def partition(nonregular):
    part1 = nonregular.subgraph([0, 1, 2])
    part2 = nonregular.subgraph([3, 4, 5])
    return Partition({0: part1, 1: part2})
