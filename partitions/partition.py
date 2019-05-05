import pandas


class Partition:
    def __init__(self, parts, data=None, assignment=None):
        if data is None:
            data = pandas.DataFrame(index=list(parts.keys()))
        if assignment is None:
            assignment = {
                node: key for key, part in parts.items() for node in part.image
            }
        self.parts = parts
        self.data = data
        self.assignment = assignment

    def __repr__(self):
        return "<Partition [{}]>".format(len(self))

    def __len__(self):
        return len(self.parts)

    def __iter__(self):
        return iter(self.parts.values())

    def __getitem__(self, i):
        return self.parts[i]

    @property
    def index(self):
        return self.data.index

    def keys(self):
        return self.parts.keys()

    def copy(self):
        return self.__class__(
            self.parts.copy(), self.data.copy(), self.assignment.copy()
        )

    def update(self, partition):
        self.parts.update(partition.parts)
        if len(self.data.columns) > 0:
            self.data.update(partition.data)
        self.assignment.update(partition.assignment)

    # These part-based methods might not work so well when we need to include
    # cut edges, unless we keep the original graph around another way.
    def with_updated_parts(self, partition):
        """Returns a new partition by updating the parts of ``self``
        with the parts of ``partition``. All of the parts of ``self``
        with keys not in ``partition.keys()`` are included in the new
        partition unchanged.

        :param Partition partition:
        """
        updated = self.copy()
        updated.update(partition)
        return updated

    def reindex(self, new_keys, in_place=False):
        reindexed_parts = {
            new_keys.get(key, key): part for key, part in self.parts.items()
        }
        reindexed_assignment = {
            node: new_keys.get(key, key) for node, key in self.assignment.items()
        }
        if in_place is True:
            self.parts = reindexed_parts
            self.data.set_index(self.data.index.map(new_keys), inplace=True)
        else:
            return self.__class__(
                reindexed_parts,
                self.data.set_index(self.data.index.map(new_keys)),
                reindexed_assignment,
            )

    @classmethod
    def from_assignment(cls, graph, assignment):
        """This creates a Partition based on the given ``assignment``
        of nodes to parts. This is analogous to (and implemented with)
        a pandas groupby operation.
        """
        grouped = graph.data.groupby(assignment)
        parts = {key: graph.subgraph(nodes) for key, nodes in grouped.groups.items()}
        return cls(
            parts,
            data=grouped.agg(graph.agg),
            assignment=assignment if isinstance(assignment, dict) else None,
        )
