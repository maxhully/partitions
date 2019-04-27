# graphs

## Project goals

-   Compatibility with the pandas/numpy/scipy stack
-   Keep the door open for extending with Cython
-   Present an intuitive interface
-   Prioritize efficiency in "big" operations (like generating spanning trees)
    over "small" and mutable operations (like adding an edge to the graph)

## Graphs

```python
>>> import pandas
>>> from graphs import Graph
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
>>> graph.neighbors[0]
array([1, 2], dtype=int32)

```

### Subgraph

```python
>>> subgraph = graph.subgraph({1, 2})
>>> subgraph.data["population"]
0    200
1     50
Name: population, dtype: int64

>>> list(subgraph.edges)
[(0, 1)]
>>> subgraph
<EmbeddedGraph [2 nodes]>
>>> subgraph.embedding
array([1, 2], dtype=int64)

>>> subgraph.embedding[0]
1

>>> subgraph.embedding[[0, 1]]
array([1, 2], dtype=int64)

```

## Partitioning a graph

```python
>>> from graphs import Partition
>>> partition = Partition.from_assignment(graph, {0: 0, 1: 1, 2: 1})
>>> partition
<Partition [2]>
>>> for part in partition:
...     print(set(part.embedding[part.nodes]))
{0}
{1, 2}

>>> partition.data["population"]
0    100
1    250
Name: population, dtype: int64

# >>> set(partition["a"].cut_edges)
# {(0, 1), (0, 2)}
# >>> set(partition.cut_edges["a", "b"])
# {(0, 1), (0, 2)}
```
