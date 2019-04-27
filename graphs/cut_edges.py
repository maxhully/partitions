from numpy import isin
from pandas import MultiIndex


def cut_edges(edges, assignment):
    row, col = edges.matrix.nonzero()
    is_cut = assignment[row] != assignment[col]
    return zip(row[is_cut], col[is_cut])


def cut_edges_for_subset(graph, subset):
    indices, neighbors = graph.neighbors.matrix[subset].nonzero()
    nodes = subset[indices]
    indicator = isin(neighbors, subset, invert=True)
    return MultiIndex.from_arrays([nodes[indicator], neighbors[indicator]])
