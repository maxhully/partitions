# graphs

## Graphs

```python
>>> import pandas
>>> from graphs import Graph
>>>
>>> graph = Graph.from_edges(
...     [(0, 1), (1, 2), (0, 2)],
...     data=pandas.DataFrame(
...         {"population": [100, 200, 50], "votes": [50, 60, 40]},
...         index=[0, 1, 2]
...     )
... )
>>> graph
<Graph ['population', 'votes']>

```

### Nodes and edges

```python
graph
>>> set(graph.nodes) == {0, 1, 2}
True
>>> set(graph.edges) == {(0, 1), (1, 2), (0, 2)}
True
>>> list(graph)
[0, 1, 2]
>>> len(graph.nodes)
3
>>> len(graph.edges)
3

```

#### Edges are undirected

```python
>>> (0, 1) in graph.edges
True
>>> (1, 0) in graph.edges
True

```

### Data

```python
>>> graph.data["population"]
0    100
1    200
2     50
Name: population, dtype: int64
>>> graph.data
   population  votes
0         100     50
1         200     60
2          50     40

```

### Neighbors

```python
>>> set(graph.neighbors[0]) == {1, 2}
True
>>> set(graph.neighbors[1]) == {0, 2}
True

```

## Partitioning a graph

```python
# >>> partition = graph.partition({0: "a", 1: "b", 2: "b"})
# >>> partition
# <Partition [2]>
# >>> for part in partition:
# ...     print(set(part.nodes))
# {0}
# {1, 2}
# >>> partition["population"]
# a    100
# b    250
# dtype: int64
# >>> set(partition["a"].cut_edges)
# {(0, 1), (0, 2)}
# >>> set(partition.cut_edges["a", "b"])
# {(0, 1), (0, 2)}
```
