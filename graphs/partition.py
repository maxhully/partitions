import pandas
from collections.abc import Mapping


class Partition(Mapping):
    def __init__(self, parts, data=None):
        if data is None:
            data = pandas.DataFrame(index=pandas.RangeIndex(start=0, stop=len(parts)))

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

    # These part-based methods might not work so well when we need to include
    # cut edges, unless we keep the original graph around another way.
    def with_updated_parts(self, partition):
        """Returns a new partition by updating the parts of this partitoin
        with the parts of ``partition``. All of the parts of this partition
        with keys not in ``partition.keys()`` are included in the new
        partition unchanged.

        :param Partition or dict partition:
        """
        if isinstance(partition, dict):
            partition = Partition.from_parts(partition)

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
        return Partition(updated_parts, data=updated_data)

    @classmethod
    def from_assignment(cls, graph, assignment):
        """This creates a Partition based on the given ``assignment``
        of nodes to parts. This is analogous to (and implemented with)
        a pandas groupby operation.
        """
        grouped = graph.data.groupby(assignment)
        parts = {key: graph.subgraph(nodes) for key, nodes in grouped.groups.items()}
        return cls(parts, data=grouped.sum())

    @classmethod
    def from_parts(cls, parts, data=None):
        """Create a Partition from an iterable or dictionary of parts.
        """
        if not isinstance(parts, dict):
            parts = dict(enumerate(parts))

        if data is None:
            keys, values = zip(*parts.items())
            data = pandas.DataFrame.from_records(
                (part.data.sum() for part in values), index=keys
            )
        elif len(data.index) != len(parts):
            raise IndexError("Partition data must be indexed by its parts")
        return cls(parts, data)
