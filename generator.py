import math

import networkx as nx
import numpy as np

from network import Network
from node import Node


def generate_nodes(n: int, k: int, h: int, random_gen=np.random.rand, split_deviation=0.0):
    tree = nx.minimum_spanning_tree(nx.random_tree(k))
    # tree = nx.balanced_tree(math.ceil(k ** (1 / h)), h)
    height = max(nx.shortest_path_length(tree, source=0).values())

    data = random_gen(n)

    split_ind = np.linspace(0, n, k + 1)[1: -1] + np.random.normal(0.0, split_deviation)
    split_ind = split_ind.astype(int)

    data_split = np.split(data, split_ind)

    bfs = list(nx.bfs_tree(tree, source=0).edges())

    network = Network()
    root = Node(network, data_split[0])
    nodes = {0: root}

    for start, end in bfs[:k-1]:
        nodes[end] = nodes[start].add_child(data_split[end])

    return network, root, data, height
