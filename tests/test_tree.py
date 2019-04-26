from graphs.tree import random_spanning_tree


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
