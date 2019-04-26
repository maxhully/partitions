from collections import Container, Sequence
from itertools import chain

import numpy
import pandas
from scipy.sparse import csr_matrix, dok_matrix, triu


class Edges(Container):
    """
    :ivar matrix: Upper-triangular adjacency matrix
    :vartype matrix: :class:`scipy.sparse.csr_matrix`
    """

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


class Neighbors(Sequence):
    """
    :ivar matrix: Symmetric adjacency matrix
    :vartype matrix: :class:`scipy.sparse.csr_matrix`
    """

    def __init__(self, matrix):
        self.matrix = (matrix + matrix.T).tocsr()

    def __repr__(self):
        return "<Neighbors [{} nodes]>".format(len(self))

    def __getitem__(self, node):
        return self.matrix.getrow(node).nonzero()[1]

    def __len__(self):
        return self.matrix.shape[0]

    def degrees(self):
        return self.matrix.indptr[1:] - self.matrix.indptr[:-1]


class Graph:
    """
    :ivar Edges edges:
    :ivar Neighbors neighbors:
    :ivar pandas.Dataframe data:
    """

    def __init__(self, matrix, data=None):
        if data is None:
            size = matrix.shape[0]
            data = pandas.DataFrame(index=pandas.RangeIndex(start=0, stop=size))

        if matrix.shape != (len(data.index), len(data.index)):
            raise IndexError("Graph data must be indexed by the graph's nodes")

        self.matrix = matrix
        self.data = data
        self.edges = Edges(matrix)
        self.neighbors = Neighbors(matrix)

    def __repr__(self):
        return "<Graph {}>".format(list(self.data.columns))

    def __iter__(self):
        return iter(self.nodes)

    def __len__(self):
        return len(self.data.index)

    def subgraph(self, nodes):
        nodes = list(nodes)

        matrix = subgraph_matrix(self.neighbors.matrix, nodes)

        subgraph_data = self.data.loc[nodes]
        subgraph_data.reset_index(inplace=True)
        node_mapping = subgraph_data.pop("index")

        return EmbeddedGraph(matrix, node_mapping, data=subgraph_data)

    @property
    def nodes(self):
        return self.data.index

    @classmethod
    def from_edges(cls, edges, data=None):
        edges = list(edges)

        if data is None:
            size = max(chain(*edges)) + 1
        else:
            size = len(data)

        matrix = dok_matrix((size, size))
        for node, neighbor in edges:
            matrix[node, neighbor] = 1
            matrix[neighbor, node] = 1

        return cls(matrix, data)


# I'm not sure about this part of the interface. It seems like we should
# represent this relationship with a GraphEmbedding object, and not
# subclass Graph. But who owns the embedding?
class EmbeddedGraph(Graph):
    def __init__(self, matrix, node_mapping, data=None):
        super().__init__(matrix, data)
        self.embedding = node_mapping.rename("embedding")

    def __repr__(self):
        return "<EmbeddedGraph [{} nodes]>".format(len(self))


def subgraph_matrix(matrix, nodes):
    subgraph_size = len(nodes)
    nodes = numpy.asarray(nodes)
    transformation = csr_matrix(
        (numpy.ones(subgraph_size), (nodes, numpy.arange(subgraph_size))),
        shape=(matrix.shape[0], subgraph_size),
    )
    return transformation.T @ matrix @ transformation
