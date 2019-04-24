from scipy.sparse import dok_matrix, csr_matrix, triu
from scipy.sparse.csgraph import minimum_spanning_tree
from itertools import chain
import pandas
import numpy


class Edges:
    def __init__(self, matrix):
        self.matrix = triu(matrix, format="csr")

    def __repr__(self):
        return "<Edges {}>".format(list(self))

    def __contains__(self, edge):
        i, j = edge
        try:
            return bool(self.matrix[i, j]) or bool(self.matrix[j, i])
        except IndexError:
            return False

    def __iter__(self):
        row, col = self.matrix.nonzero()
        return zip(row, col)

    def __len__(self):
        return self.matrix.count_nonzero()


class Neighbors:
    def __init__(self, matrix):
        self.matrix = matrix

    def __repr__(self):
        return "<Neighbors [{} nodes]>".format(self.matrix.shape[0])

    def __getitem__(self, node):
        return self.matrix.getrow(node).nonzero()[1]


class Graph:
    def __init__(self, matrix, data=None):
        if data is None:
            size = matrix.shape[0]
            data = pandas.DataFrame(index=pandas.RangeIndex(start=0, stop=size))

        if matrix.shape != (len(data.index), len(data.index)):
            raise IndexError("Graph data must be indexed by the graph's nodes")

        self.matrix = matrix.tocsr()
        self.data = data
        self.edges = Edges(matrix)
        self.neighbors = Neighbors(matrix)

    def __repr__(self):
        return "<Graph {}>".format(list(self.data.columns))

    def __iter__(self):
        return iter(self.nodes)

    @property
    def nodes(self):
        return self.data.index

    @classmethod
    def from_edges(cls, edges, data=None):
        if data is None:
            size = max(chain(*edges)) + 1
        else:
            size = len(data)

        matrix = dok_matrix((size, size))
        for node, neighbor in edges:
            matrix[node, neighbor] = 1
            matrix[neighbor, node] = 1

        return cls(matrix.tocsr(), data)


def random_spanning_tree(graph):
    row_indices, col_indices = graph.matrix.nonzero()
    weights = numpy.random.random(len(row_indices))
    weighted_matrix = csr_matrix((weights, (row_indices, col_indices)))
    tree = minimum_spanning_tree(weighted_matrix)
    tree += tree.T
    return Graph(tree.astype(bool), data=graph.data)
