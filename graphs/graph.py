from itertools import chain

import numpy
import pandas
from scipy.sparse import csr_matrix, dok_matrix

from .cut_edges import Boundary
from .edges import Edges
from .neighbors import Neighbors


class Graph:
    """
    :ivar Edges edges:
    :ivar Neighbors neighbors:
    :ivar pandas.DataFrame data:
    :ivar scipy.sparse.csr_matrix matrix:
    """

    def __init__(self, matrix, data=None, edge_data=None):
        self.matrix = matrix
        self.data = data
        self.edges = Edges(matrix, data=edge_data)
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
    def from_matrix(cls, matrix, data=None, edge_data=None):
        """Create a graph from an adjacency matrix. The matrix can have
        any ``dtype`` and need not be symmetric---it will be symmetrized and
        cast to ``bool`` before instantiating the graph.
        """
        matrix = csr_matrix(matrix.astype(bool, copy=False), copy=False)
        matrix += matrix.T

        if data is None:
            data = pandas.DataFrame(
                index=pandas.RangeIndex(start=0, stop=matrix.shape[0])
            )

        if matrix.shape != (len(data.index), len(data.index)):
            raise IndexError("Graph data must be indexed by the graph's nodes")

        return cls(matrix, data=data, edge_data=edge_data)

    @classmethod
    def from_edges(cls, edges, data=None):
        """Create a graph from an iterable of edges.

        >>> graph = Graph.from_edges([(0, 1), (1, 2), (2, 3)])
        >>> assert set(graph.edges) == {(0, 1), (1, 2), (2, 3)}
        >>> assert set(graph.nodes) == {0, 1, 2, 3}
        """
        if isinstance(edges, pandas.DataFrame):
            edge_data, edges = edges, edges.index
            first = edges.get_level_values(0)
            second = edges.get_level_values(1)
            if not (first <= second).all():
                raise ValueError("edge data indices (i, j) must satisfy i <= j")
        else:
            edges = list(edges)
            edge_data = None

        if data is None:
            size = max(chain(*edges)) + 1
        else:
            size = len(data)

        matrix = dok_matrix((size, size))
        for node, neighbor in edges:
            matrix[node, neighbor] = 1

        return cls.from_matrix(matrix, data, edge_data)


class EmbeddedGraph(Graph):
    """
    :ivar Edges edges:
    :ivar Neighbors neighbors:
    :ivar pandas.DataFrame data:
    :ivar scipy.sparse.csr_matrix matrix:

    :ivar numpy.ndarray image: the image of this graph's nodes in the graph
        where this graph is embedded. That is, node ``i`` in this graph
        corresponds to node ``image[i]`` in the graph where this node is embedded.
    :ivar Boundary boundary: the boundary nodes, edges, and data of the
        embedded graph.
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
        self.boundary = Boundary(graph, image)

    def __repr__(self):
        return "<EmbeddedGraph [{} nodes]>".format(len(self))

    def subgraph(self, nodes):
        """Given a subset of nodes, returns the subgraph induced by those nodes.
        Since we are already an EmbeddedGraph, we return an EmbeddedGraph of
        the same graph (so that there are no subgraphs-of-subgraphs).
        :param numpy.ndarray or iterable nodes:
        :rtype EmbeddedGraph:
        """
        if not isinstance(nodes, numpy.ndarray):
            nodes = numpy.asarray(list(nodes))
        return EmbeddedGraph(self.graph, self.image[nodes])

    @property
    def cut_edges(self):
        """All of the edges in the ambient graph that have one node in this
        graph and one outside of it.
        :rtype: pandas.MultiIndex
        """
        return self.boundary.cut_edges


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
