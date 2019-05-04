import pandas

from .graph import Part


class Partition:
    part_class = Part

    def __init__(self, parts, data=None):
        if data is None:
            data = pandas.DataFrame(index=list(parts.keys()))
        self.parts = parts
        self.data = data

    def __repr__(self):
        return "<Partition [{}]>".format(len(self))

    def __len__(self):
        return len(self.parts)

    def __iter__(self):
        return iter(self.parts.values())

    def __getitem__(self, i):
        return self.parts[i]

    def keys(self):
        return self.parts.keys()

    # These part-based methods might not work so well when we need to include
    # cut edges, unless we keep the original graph around another way.
    def with_updated_parts(self, partition):
        """Returns a new partition by updating the parts of this partitoin
        with the parts of ``partition``. All of the parts of this partition
        with keys not in ``partition.keys()`` are included in the new
        partition unchanged.

        :param Partition partition:
        """
        updated_parts = self.parts.copy()
        updated_parts.update(partition.parts)

        # Updating an empty DataFrame raises an exception, so we make sure that
        # we have data to update.
        if len(self.data.columns) > 0:
            updated_data = self.data.copy()
            updated_data.update(partition.data)
        else:
            # If our data is empty, we stay empty.
            updated_data = self.data

        return Partition(updated_parts, updated_data)

    def reindex(self, new_keys, in_place=False):
        reindexed_parts = {new_keys[key]: part for key, part in self.parts.items()}
        if in_place is True:
            self.parts = reindexed_parts
            self.data.set_index(self.data.index.map(new_keys), inplace=True)
        else:
            return Partition(
                reindexed_parts, self.data.set_index(self.data.index.map(new_keys))
            )

    @classmethod
    def from_assignment(cls, graph, assignment):
        """This creates a Partition based on the given ``assignment``
        of nodes to parts. This is analogous to (and implemented with)
        a pandas groupby operation.
        """
        grouped = graph.data.groupby(assignment)
        parts = {
            key: graph.subgraph(nodes, subgraph_class=cls.part_class)
            for key, nodes in grouped.groups.items()
        }
        return cls(parts, data=grouped.agg(graph.agg))
