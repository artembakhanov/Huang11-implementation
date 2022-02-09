from node import Node, Network, ranks
import numpy as np

n = 100_000
k = 4
data = np.random.rand(n)
split_ind = np.linspace(0, n, k + 1)[1: -1] + np.random.rand(k - 1) * n // 2
split_ind = split_ind.astype(int)
data1, data2, data3, data4 = np.split(data, split_ind)

network = Network(verbose=False)
node_root = Node(network, data1)
node1 = node_root.add_child(data2)
node2 = node_root.add_child(data3)
node3 = node_root.add_child(data4)
# node = Node(network, np.array([1, 1, 1, 1, 1]))

data = np.sort(data)
i = 1000
print("orig:", data[i])

# node_root.new_task(0.4, 0.5)
# print(node_root.rank_query(i))
# print(np.where(data == node_root.rank_query(i)))
# print(network.stat)
# network.reset()

for eps in np.arange(0.001, 0.5, 0.05):
    node_root.new_task(0.4, eps)
    print("res: ", node_root.rank_query(i))
    print("res ind:", np.where(data == node_root.rank_query(i)))
    print("stat:", network.stat)

    network.reset()

# print(node_root)
# print(node1)
# print(node_root._task)
# print(node_root.global_rank(6))
# print(node_root._global_ranks)
# print(node_root.rank_query(6))
# print(ranks([1, 1, 2, 3, 4, 5, 6, 7, 8, 8]))
