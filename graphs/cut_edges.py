def cut_edges(edges, assignment):
    row, col = edges.matrix.nonzero()
    is_cut = assignment[row] != assignment[col]
    return zip(row[is_cut], col[is_cut])


def cut_edges_for_subset(graph, subset, assignment):
    indices, neighbors = graph.neighbors.matrix[subset].nonzero()
    nodes = subset[indices]
    indicator = assignment[nodes] != assignment[neighbors]
    return zip(nodes[indicator], neighbors[indicator])
