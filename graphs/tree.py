from collections import deque

import numpy
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree, breadth_first_order

from .graph import Graph
from .partition import Partition


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

    return Partition.from_assignment(graph, assignment)


def random_cut_edge(partition):
    keys = numpy.array(partition.keys())
    weights = numpy.array([len(part.cut_edges) for part in partition])
    part_key = numpy.random.choice(keys, p=weights)
    edge = numpy.random.choice(partition[part_key].cut_edges)

