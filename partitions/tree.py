from collections import deque
from itertools import repeat, chain
from numbers import Number

import numpy
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree, breadth_first_order

from .graph import Graph


def random_spanning_tree(graph):
    row_indices, col_indices = graph.matrix.nonzero()
    weights = numpy.random.random(len(row_indices))
    weighted_matrix = csr_matrix((weights, (row_indices, col_indices)))
    tree = minimum_spanning_tree(weighted_matrix)
    return Graph.from_matrix(tree, data=graph.data)


def contract_leaves_until_balanced_or_None(
    tree, population, pop_bounds, choice=numpy.random.choice
):
    degrees = tree.neighbors.degrees()
    pops = population.copy()
    lower, upper = pop_bounds
    # Every node begins assigned to itself.
    assignment = tree.nodes.to_numpy(copy=True)

    root = choice(tree.nodes[degrees > 1])
    leaves = deque(tree.nodes[degrees == 1])

    _, pred = breadth_first_order(tree.neighbors.matrix, root, return_predecessors=True)

    while len(leaves) > 0:
        leaf = leaves.popleft()
        if (lower < pops[leaf]) and (pops[leaf] < upper):
            return assignment == leaf
        # Contract the leaf:
        parent = pred[leaf]
        pops[parent] += pops[leaf]
        assignment[assignment == leaf] = parent
        degrees[parent] -= 1

        if degrees[parent] == 1 and parent != root:
            leaves.append(parent)

    return None


def bipartition_tree(graph, population, bounds):
    assignment = None
    while assignment is None:
        tree = random_spanning_tree(graph)
        assignment = contract_leaves_until_balanced_or_None(tree, population, bounds)
    return assignment


def recursive_partition(
    graph, number_of_parts, population, pop_bounds, method=bipartition_tree
):
    assignment = numpy.zeros_like(graph.nodes)
    remaining = graph
    # Count down, because we'll leave the existing zeroes as the assignment
    # for all the remaining nodes.
    for i in range(number_of_parts)[1:]:
        in_part_i = method(remaining, population, pop_bounds)
        nodes = getattr(remaining, "image", remaining.nodes)
        assignment[nodes[in_part_i]] = i
        remaining = graph.subgraph(graph.nodes[assignment == 0])

    return assignment


def random_cut_edge(partition):
    keys = partition.index
    weights = numpy.array([len(part.cut_edges) for part in partition])
    weights = weights / weights.sum()
    part_key = numpy.random.choice(keys, p=weights)
    edge = numpy.random.choice(partition[part_key].cut_edges)
    return edge


def map_with_boolean_array(array, selector, values):
    """Returns a dictionary mapping ``array[i]`` to ``values[selector[i]]``.

    Uses :mod:`itertools` to effectively do the dictionary lookups in NumPy.
    """
    true_value, false_value = values[True], values[False]
    return dict(
        chain(
            zip(array[selector], repeat(true_value)),
            zip(array[numpy.invert(selector)], repeat(false_value)),
        )
    )


class ReCom:
    def __init__(self, pop_column, bounds, method=bipartition_tree):
        if len(bounds) != 2:
            raise TypeError(
                "population bounds must be a tuple 2 numbers (upper and lower bound)"
            )
        if not all(isinstance(bound, Number) for bound in bounds):
            raise TypeError("lower and upper bounds must be Numbers")
        if bounds[0] >= bounds[1]:
            raise TypeError(
                "lower bound (bounds[0]) must be less than upper bound (bounds[1])"
            )
        if not isinstance(pop_column, str):
            raise TypeError(
                "pop_column should be the string name of the population column"
            )
        if not callable(method):
            raise TypeError("the partitioning method should be callable")

        self.pop_column = pop_column
        self.bounds = bounds
        self.method = method

    def __call__(self, partition):
        i, j = random_cut_edge(partition)
        p, q = partition.assignment[i], partition.assignment[j]
        subgraph = partition[p].union(partition[q], disjoint=True)

        population = subgraph.data[self.pop_column]

        assignment_array = self.method(subgraph, population, self.bounds)
        assignment = map_with_boolean_array(
            subgraph.image, assignment_array, {False: p, True: q}
        )

        # It is important to use subgraph.graph rather than just subgraph, so that
        # the assignment lines up properly with graph.data and graph.nodes.
        new_parts = partition.__class__.from_assignment(subgraph.graph, assignment)
        return partition.with_updated_parts(new_parts)
