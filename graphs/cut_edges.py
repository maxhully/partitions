def cut_edges(edges, assignment):
    row, col = edges.matrix.nonzero()
    is_cut = assignment[row] != assignment[col]
    return zip(row[is_cut], col[is_cut])
