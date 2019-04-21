from scipy.sparse import dok_matrix
from numpy import nonzero


class EdgeSet:
    def __init__(self, edges):
        self.edges = set(tuple(sorted(edge)) for edge in edges)

    def __contains__(self, edge):
        return tuple(sorted(edge)) in self.edges

    def __iter__(self):
        return iter(self.edges)


class Neighbors:
    def __init__(self, index, pairs):
        self.index = index
        matrix = dok_matrix((len(index), len(index)))
        for node, neighbor in pairs:
            i, j = index.get_loc(node), index.get_loc(neighbor)
            matrix[i, j] = 1
            matrix[j, i] = 1
        self._matrix = matrix.tocsr()

    def __getitem__(self, node):
        i = self.index.get_loc(node)
        indices = self._matrix.getrow(i).nonzero()[1]
        return self.index[indices]


class Graph:
    def __init__(self, nodes, edges, data):
        if set(nodes) != set(data.index):
            raise IndexError("Graph data must be indexed by the graph's nodes")
        self.nodes = set(nodes)
        self.edges = set(edges)
        self.data = data

        self.neighbors = Neighbors(data.index, edges)

    def __repr__(self):
        return "<Graph {}>".format(list(self.data.columns))
