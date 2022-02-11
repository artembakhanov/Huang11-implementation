import math

import networkx as nx
import numpy as np

from network import Network
from node import Node
from random import randint, choice


def generate_tree(k: int, h: int = None):
    if h is None:
        return nx.minimum_spanning_tree(nx.random_tree(k))

    if h > k - 1:
        raise Exception("Such tree does not exist")

    node = h
    paths = [list(range(0, h))]

    while node < k - 1:
        path = choice(paths)
        ind = randint(1, min(len(path), h - 1))
        paths.append(path[:ind] + [node])
        node += 1

    tree = nx.prefix_tree(paths)
    tree.remove_node(-1)
    tree = tree.to_undirected()

    return tree


def generate_nodes(n: int, k: int, h: int = None, random_gen=np.random.rand, split_deviation=0.0):
    tree = generate_tree(k, h)
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

    for start, end in bfs[:k - 1]:
        if end == -1:
            continue
        nodes[end] = nodes[start].add_child(data_split[end])

    return network, root, data, height
