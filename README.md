# graphs

```python
>>> import pandas
>>> from graphs import Graph
>>>
>>> graph = Graph(
...     nodes=[1, 2, 3],
...     edges=[(1, 2), (2, 3), (1, 3)]
...     data=pandas.DataFrame({
...         "population": [100, 200, 50],
...         "votes": [50, 60, 40]
...     })
... )
<Graph ["population", "votes"]>
>>> set(graph.nodes)
{1, 2, 3}
>>> graph["population"]
1    100
2    200
3    50
dtype: int64
>>> graph.data
  population votes
1 100        50
2 200        60
3 50         40
>>> partition = graph.partition({1: "a", 2: "b", 3: "b"})
>>> partition
<Partition [2]>
>>> for part in partition:
...     print(set(part.nodes))
{1}
{2, 3}
>>> partition["population"]
a    100
b    250
dtype: int64
>>> set(partition["a"].cut_edges)
{(1, 2), (1, 3)}
>>> set(partition.cut_edges["a", "b"])
{(1, 2), (1, 3)}
```
