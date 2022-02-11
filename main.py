from generator import generate_nodes
from node import SuperNode, Network, ranks
import numpy as np

n = 10_000_000
k = 8196

network, node_root, data, height = generate_nodes(n, k, 1000, split_deviation=0.0, random_gen=np.random.standard_normal,
                                                  graph_partition=True)
print(f"height: {height}")
# data = np.sort(data)

q = 0.25
i = q * n
per = np.percentile(data, q * 100)
print("orig rank:", i)
print("orig:", per)

# node_root.new_task(0.4, 0.5)
# print(node_root.rank_query(i))
# print(np.where(data == node_root.rank_query(i)))
# print(network.stat)
# network.reset()

eps = 0.0001
print(f"max e should be {eps * n}")

node_root.new_task(eps)
print("res: ", node_root.rank_query(i))
print("res rank:", node_root.rank_query(i, return_rank=True)[1])
print("stat:", network.stat)

# for eps in np.logspace(-4, -2, 50):
#     node_root.new_task(eps)
#     print(node_root._task._p)
#     print("res: ", node_root.rank_query(i))
#     print("res ind:", np.where(data == node_root.rank_query(i)))
#     print("stat:", network.stat)
#
#     network.reset()

# print(node_root)
# print(node1)
# print(node_root._task)
# print(node_root.global_rank(6))
# print(node_root._global_ranks)
# print(node_root.rank_query(6))
# print(ranks([1, 1, 2, 3, 4, 5, 6, 7, 8, 8]))
