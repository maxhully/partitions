from graphs import Graph, Neighbors
import pandas
import pytest


class TestGraph:
    def test_init(self):
        graph = Graph(
            nodes=[1, 2, 3],
            edges=[(1, 2), (2, 3), (1, 3)],
            data=pandas.DataFrame(
                {"population": [100, 200, 50], "votes": [50, 60, 40]}, index=[1, 2, 3]
            ),
        )
        assert graph

    def test_data_index_must_match_graph(self):
        with pytest.raises(IndexError):
            Graph(
                nodes=[1, 2, 3],
                edges=[(1, 2), (2, 3), (1, 3)],
                data=pandas.DataFrame(
                    {"population": [100, 200, 50], "votes": [50, 60, 40]},
                    index=[0, 1, 2],
                ),
            )


class TestNeighbors:
    def test_init(self):
        neighbors = Neighbors(pandas.Index([0, 1, 2]), [(0, 1), (1, 2), (2, 0)])
        assert set(neighbors[0]) == {1, 2}

    def test_index(self):
        neighbors = Neighbors(
            pandas.Index(list("abcd")), [("a", "b"), ("b", "c"), ("c", "d"), ("d", "a")]
        )
        assert set(neighbors["a"]) == {"b", "d"}
        assert set(neighbors["b"]) == {"a", "c"}
        assert set(neighbors["c"]) == {"b", "d"}
        assert set(neighbors["d"]) == {"a", "c"}
        for node in "abcd":
            assert len(neighbors[node]) == 2
