import pandas
from collections.abc import Sequence


class Partition(Sequence):
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

    def with_updated_parts(self, new_parts):
        updated_parts = self.parts.copy()
        updated_parts.update(new_parts)

        if len(self.data) > 0:
            updated_data = self.data.copy()
            for i, part in new_parts.items():
                updated_data.loc[i] = part.data.sum()
        else:
            updated_data = self.data
        return Partition(updated_parts, data=updated_data)

    @classmethod
    def from_assignment(cls, graph, assignment):
        grouped = graph.data.groupby(assignment)
        node_sets = grouped.groups.values()
        parts = dict(enumerate(graph.subgraph(nodes) for nodes in node_sets))
        return cls(parts, data=grouped.sum())

    @classmethod
    def from_parts(cls, parts, data=None):
        if not isinstance(parts, dict):
            parts = dict(enumerate(parts))

        if data is None:
            data = pandas.DataFrame.from_records(
                part.data.sum() for part in parts.values()
            )
        elif len(data.index) != len(parts):
            raise IndexError("Partition data must be indexed by its parts")
        return cls(parts, data)
