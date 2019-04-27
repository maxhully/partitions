from collections import Sequence


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
