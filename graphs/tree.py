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
        print(leaf)
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
    return Partition.from_assignment(graph, assignment)
