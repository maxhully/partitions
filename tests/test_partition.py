import numpy
import pandas
import pytest

from graphs.partition import Partition


@pytest.fixture
def partition(nonregular):
    part1 = nonregular.subgraph([0, 1, 2])
    part2 = nonregular.subgraph([3, 4, 5])
    return Partition({0: part1, 1: part2})


def check_embedding_of_parts(partition, expected_nodes):
    """Check that the parts of the partition embed into the
    set of ``expected_nodes`` and comprise the entire set.
    """
    re_embedded_nodes = set()
    for part in partition:
        re_embedded_nodes.update(part.image[part.nodes])
    return re_embedded_nodes == set(expected_nodes)


class TestPartition:
    def test_from_assignment(self, k4):
        assignment = numpy.asarray([0, 0, 1, 1])
        partition = Partition.from_assignment(k4, assignment)
        assert len(partition) == 2
        assert check_embedding_of_parts(partition, k4.nodes)

    def test_from_assignment_allows_any_indices(self, k4):
        assignment = numpy.asarray(["a", "a", "b", "b"])
        partition = Partition.from_assignment(k4, assignment)
        assert set(partition["a"].image) == {0, 1}
        assert set(partition["b"].image) == {2, 3}

    def test_from_parts(self, nonregular):
        nonregular.data = pandas.DataFrame({"data1": [100, 200, 100, 300, 400, 300]})
        part1 = nonregular.subgraph([0, 1, 2])
        part2 = nonregular.subgraph([3, 4, 5])
        partition = Partition.from_parts([part1, part2])
        assert len(partition) == 2
        assert check_embedding_of_parts(partition, {0, 1, 2, 3, 4, 5})

        # Check aggregated data is correct
        assert (
            partition.data["data1"] == pandas.Series([400, 1000], index=[0, 1])
        ).all()

    def test_from_parts_with_custom_data(self, nonregular):
        data = pandas.DataFrame({"test_data": [5, 10]})

        part1 = nonregular.subgraph([0, 1, 2])
        part2 = nonregular.subgraph([3, 4, 5])
        partition = Partition.from_parts([part1, part2], data=data)
        assert list(partition.data["test_data"]) == [5, 10]

    def test_from_parts_raises_for_incorrectly_indexed_data(self, nonregular):
        part1 = nonregular.subgraph([0, 1])
        part2 = nonregular.subgraph([2, 3])
        part3 = nonregular.subgraph([4, 5])
        data = pandas.DataFrame({"test_data": [1, 2]})
        with pytest.raises(IndexError):
            Partition.from_parts([part1, part2, part3], data=data)

    def test_data_in_from_assignment(self, k4):
        k4.data = pandas.DataFrame({"test_data": [20, 10, 500, 100]})
        assignment = numpy.asarray([0, 1, 1, 0])
        partition = Partition.from_assignment(k4, assignment)
        assert (
            partition.data["test_data"] == pandas.Series([120, 510], index=[0, 1])
        ).all()

    def test_getitem(self, nonregular):
        part1 = nonregular.subgraph([0, 1, 2])
        part2 = nonregular.subgraph([3, 4, 5])
        partition = Partition.from_parts([part1, part2])
        assert partition[0] is part1
        assert partition[1] is part2

    def test_repr(self, partition):
        assert repr(partition) == "<Partition [2]>"

    def test_can_immutably_update_parts(self, nonregular):
        part1 = nonregular.subgraph([0, 1])
        part2 = nonregular.subgraph([2, 3])
        part3 = nonregular.subgraph([4, 5])
        partition = Partition.from_parts([part1, part2, part3])

        updates = {1: nonregular.subgraph([2, 4]), 2: nonregular.subgraph([3, 5])}

        new_partition = partition.with_updated_parts(updates)

        check_embedding_of_parts(new_partition, nonregular.nodes)
        assert new_partition[1] is updates[1]
        assert new_partition[2] is updates[2]
        assert new_partition[0] is partition[0]

    def test_updating_updates_data(self, nonregular):
        data = pandas.DataFrame({"test_data": [50, 100, 60, 120, 33, 66]})
        nonregular.data = data
        part1 = nonregular.subgraph([0, 1])
        part2 = nonregular.subgraph([2, 3])
        part3 = nonregular.subgraph([4, 5])
        partition = Partition.from_parts([part1, part2, part3])

        updates = {1: nonregular.subgraph([2, 4]), 2: nonregular.subgraph([3, 5])}

        new_partition = partition.with_updated_parts(updates)
        assert new_partition.data["test_data"][1] == 93
        assert new_partition.data["test_data"][2] == 186

    def test_updating_empty_partition_does_not_copy_data(self, partition):
        updated = partition.with_updated_parts(partition)
        assert updated.data is partition.data
