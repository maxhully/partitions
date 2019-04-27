from collections import Container, Sequence
from itertools import chain

import numpy
import pandas
from scipy.sparse import csr_matrix, dok_matrix, triu

from .cut_edges import cut_edges_for_subset


class Edges(Container):
    """
    :ivar matrix: Upper-triangular adjacency matrix
    :vartype matrix: :class:`scipy.sparse.csr_matrix`
    """

    def __init__(self, matrix):
        """
        :param matrix: Symmetric adjacency matrix
        :type matrix: :class:`scipy.sparse.csr_matrix`
        """
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
        """
        :param matrix: Symmetric adjacency matrix
        :type matrix: :class:`scipy.sparse.csr_matrix`

        We assume the matrix has already been made symmetric. This minimizes the
        complexity of the constructor and leaves open the possibility of using
        this class for a directed graph.
        """
        self.matrix = matrix.tocsr(copy=False)

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
    :ivar pandas.DataFrame data:
    :ivar scipy.sparse.csr_matrix matrix:
    """

    def __init__(self, matrix, data=None):
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
        """Given a subset of nodes, returns the subgraph induced by those nodes.
        :param numpy.ndarray or iterable nodes:
        :rtype EmbeddedGraph:
        """
        if not isinstance(nodes, numpy.ndarray):
            nodes = numpy.asarray(list(nodes))
        return EmbeddedGraph(self, nodes)

    @property
    def nodes(self):
        return self.data.index

    @classmethod
    def from_matrix(cls, matrix, data=None):
        """Create a graph from an adjacency matrix. The matrix can have
        any ``dtype`` and need not be symmetric---it will be symmetrized and
        cast to ``bool`` before instantiating the graph.
        """
        matrix = csr_matrix(matrix.astype(bool, copy=False), copy=False)
        matrix += matrix.T

        if data is None:
            size = matrix.shape[0]
            data = pandas.DataFrame(index=pandas.RangeIndex(start=0, stop=size))

        if matrix.shape != (len(data.index), len(data.index)):
            raise IndexError("Graph data must be indexed by the graph's nodes")

        return cls(matrix, data=data)

    @classmethod
    def from_edges(cls, edges, data=None):
        """Create a graph from an iterable of edges.

        >>> graph = Graph.from_edges([(0, 1), (1, 2), (2, 3)])
        >>> assert set(graph.edges) == {(0, 1), (1, 2), (2, 3)}
        >>> assert set(graph.nodes) == {0, 1, 2, 3}
        """
        edges = list(edges)

        if data is None:
            size = max(chain(*edges)) + 1
        else:
            size = len(data)

        matrix = dok_matrix((size, size))
        for node, neighbor in edges:
            matrix[node, neighbor] = 1

        return cls.from_matrix(matrix, data)


# I'm not sure about this part of the interface. It seems like we should
# represent this relationship with a GraphEmbedding object, and not
# subclass Graph. But who owns the embedding?
class EmbeddedGraph(Graph):
    """
    :ivar numpy.ndarray image: the image of this graph's nodes in the graph
        where this graph is embedded. That is, node ``i`` in this graph
        corresponds to node ``image[i]`` in the graph where this node is embedded.
    """

    def __init__(self, graph, image):
        """
        :param Graph graph: the graph where this graph is embedded.
        :param numpy.ndarray image: the nodes in ``graph`` that this graph's
            nodes are mapped to.
        """
        matrix = subgraph_matrix(graph.matrix, image)
        data = graph.data.loc[image]
        data.reset_index(inplace=True, drop=True)

        super().__init__(matrix, data)

        self.image = image
        self.graph = graph
        self.cut_edges = cut_edges_for_subset(graph, image)

    def __repr__(self):
        return "<EmbeddedGraph [{} nodes]>".format(len(self))


def subgraph_matrix(matrix, nodes):
    """Given an adjacency matrix and list or array of nodes, returns the
    adjacency matrix of the induced subgraph.
    """
    subgraph_size = len(nodes)
    transformation = csr_matrix(
        (numpy.ones(subgraph_size), (nodes, numpy.arange(subgraph_size))),
        shape=(matrix.shape[0], subgraph_size),
    )
    return transformation.T @ matrix @ transformation
