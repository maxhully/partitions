from collections import Container

from scipy.sparse import triu


class Edges(Container):
    """
    :ivar matrix: Upper-triangular adjacency matrix
    :vartype matrix: :class:`scipy.sparse.csr_matrix`
    """

    def __init__(self, matrix, data=None):
        """
        :param matrix: Symmetric adjacency matrix
        :type matrix: :class:`scipy.sparse.csr_matrix`
        :param pandas.DataFrame or None data: edge data, indexed by a
            :class:`pandas.MultiIndex` whose indices (i, j) satisfy i <= j.
        """
        self.matrix = triu(matrix, format="csr")
        self.data = data

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
