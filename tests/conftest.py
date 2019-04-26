from graphs import Graph
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
