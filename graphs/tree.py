import numpy
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree

from .graph import Graph


def random_spanning_tree(graph):
    row_indices, col_indices = graph.matrix.nonzero()
    weights = numpy.random.random(len(row_indices))
    weighted_matrix = csr_matrix((weights, (row_indices, col_indices)))
    tree = minimum_spanning_tree(weighted_matrix)
    tree += tree.T
    return Graph(tree.astype(bool), data=graph.data)
