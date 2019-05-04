from graphs.tree import (
    random_spanning_tree,
    contract_leaves_until_balanced_or_None,
    bipartition_tree,
    recursive_partition,
    random_cut_edge,
    ReCom,
)
from graphs import Graph, Partition
from scipy.sparse.csgraph import connected_components
import numpy
import pandas


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


class TestContractEdgesUntilBalanced:
    def test_on_10x10(self):
        edges = [(i + 10 * j, i + 10 * j + 1) for i in range(9) for j in range(10)]
        edges += [(10 * j, 10 * j + 10) for j in range(9)]
        graph = Graph.from_edges(edges)
        population = numpy.ones_like(graph.nodes)
        bounds = (10, 90)

        assignment = contract_leaves_until_balanced_or_None(graph, population, bounds)
        assert len(assignment) == len(graph.nodes)
        assert len(numpy.unique(assignment)) == 2

        subgraph = graph.subgraph(graph.nodes[assignment])
        assert connected_components(subgraph.neighbors.matrix, return_labels=False) == 1

    def test_on_small(self):
        graph = Graph.from_edges([(0, 1), (1, 2)])
        population = numpy.ones_like(graph.nodes)
        bounds = (0, 3)

        assignment = contract_leaves_until_balanced_or_None(
            graph, population, bounds, choice=lambda x: 1
        )
        assert assignment[0] == assignment[1] or assignment[1] == assignment[2]
        assert len(numpy.unique(assignment)) == 2

    def test_on_medium(self):
        graph = Graph.from_edges(
            [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (5, 7), (5, 8)]
        )
        population = numpy.ones_like(graph.nodes)
        bounds = (2, 7)

        assignment = contract_leaves_until_balanced_or_None(graph, population, bounds)
        assert len(numpy.unique(assignment)) == 2
        subgraph = graph.subgraph(graph.nodes[assignment])
        assert connected_components(subgraph.neighbors.matrix, return_labels=False) == 1
        assert (2 <= population[assignment].sum()) and (
            population[assignment].sum() <= 7
        )

    def test_impossible(self):
        graph = Graph.from_edges([(0, 1), (1, 2), (2, 3)])
        population = numpy.array([1, 5, 8, 5])
        bounds = (3, 5)
        assignment = contract_leaves_until_balanced_or_None(graph, population, bounds)
        assert assignment is None


class TestBipartitionTree:
    def test_on_10x10(self):
        edges = [(i + 10 * j, i + 10 * j + 1) for i in range(9) for j in range(10)]
        edges += [(i + 10 * j, i + 10 * j + 10) for i in range(10) for j in range(9)]
        graph = Graph.from_edges(edges)
        population = numpy.ones_like(graph.nodes)
        bounds = (30, 70)

        assignment = bipartition_tree(graph, population, bounds)
        partition = Partition.from_assignment(graph, assignment)
        assert len(partition) == 2
        assert set(node for part in partition for node in part.image) == set(
            graph.nodes
        )
        for part in partition:
            assert 30 <= len(part.nodes) and len(part.nodes) <= 70

        for part in partition:
            assert connected_components(part.neighbors.matrix, return_labels=False) == 1


class TestRecursivePartition:
    def test_on_10x10(self):
        edges = [(i + 10 * j, i + 10 * j + 1) for i in range(9) for j in range(10)]
        edges += [(i + 10 * j, i + 10 * j + 10) for i in range(10) for j in range(9)]
        graph = Graph.from_edges(edges)
        population = numpy.ones_like(graph.nodes)
        ideal_pop = population.sum() / 5
        bounds = (ideal_pop * 0.8, ideal_pop * 1.2)

        assignment = recursive_partition(graph, 5, population, bounds)
        partition = Partition.from_assignment(graph, assignment)
        assert len(partition) == 5
        assert set(node for part in partition for node in part.image) == set(
            graph.nodes
        )
        # The 0th part made up of the left-over nodes often has too much population,
        # so we are only sure that parts >= 1 have the right population.
        for part in list(partition)[1:]:
            assert bounds[0] < len(part.nodes)
            assert len(part.nodes) < bounds[1]

        for part in partition:
            assert connected_components(part.neighbors.matrix, return_labels=False) == 1


def test_random_cut_edge(partition):
    assert len(partition) == 2

    i, j = random_cut_edge(partition)
    assert (i in partition[0].image and j in partition[1].image) or (
        i in partition[1].image and j in partition[0].image
    )


class TestReCom:
    def test_gives_a_partition(self, k8):
        k8.data = pandas.DataFrame({"pop": [1] * 8})
        # Allow 1-3 nodes per part
        bounds = (0.9, 3.1)
        partition = Partition.from_assignment(
            k8, dict(enumerate([0, 0, 1, 1, 2, 2, 3, 3]))
        )
        recom = ReCom("pop", bounds)
        assert len(partition) == 4

        new_partition = recom(partition)
        assert len(new_partition) == 4
        assert all(len(part) in {1, 2, 3} for part in new_partition)
        nodes = list(node for part in new_partition for node in part.image)
        assert len(nodes) == 8
        assert set(nodes) == set(k8.nodes)
