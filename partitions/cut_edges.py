import numpy
from pandas import MultiIndex


class Boundary:
    """
    :ivar numpy.ndarray nodes:
    :ivar numpy.ndarray neighbors:
    :ivar pandas.MultiIndex cut_edges:
    :ivar pandas.DataFrame edge_data:
    """

    def __init__(self, graph, subset):
        nodes, neighbors = boundary_nodes_and_neighbors(graph, subset)
        self.nodes = nodes
        self.neighbors = neighbors
        self.cut_edges = sorted_multiindex(nodes, neighbors)
        if graph.edges.data is not None:
            self.data = graph.edges.data.loc[self.cut_edges].sum()

    def unique_nodes(self):
        return numpy.unique(self.nodes)

    def unique_neighbors(self):
        return numpy.unique(self.neighbors)


def cut_edges(edges, assignment):
    row, col = edges.matrix.nonzero()
    is_cut = assignment[row] != assignment[col]
    return zip(row[is_cut], col[is_cut])


def boundary_nodes_and_neighbors(graph, subset):
    indices, neighbors = graph.neighbors.matrix[subset].nonzero()
    nodes = subset[indices]
    indicator = numpy.isin(neighbors, subset, invert=True)
    boundary_nodes = nodes[indicator]
    boundary_neighbors = neighbors[indicator]
    return boundary_nodes, boundary_neighbors


def sorted_multiindex(row, col):
    return MultiIndex.from_arrays(numpy.sort(numpy.array([row, col]), axis=0))


def cut_edges_for_subset(graph, subset):
    return sorted_multiindex(*boundary_nodes_and_neighbors(graph, subset))
