import math

import networkx as nx
import numpy as np

from helper import edges_union
from network import Network
from node import SuperNode
from random import randint, choice
from collections import defaultdict
from ortools.algorithms import pywrapknapsack_solver


def generate_tree(k: int, h: int = None):
    if h is None:
        return nx.minimum_spanning_tree(nx.random_tree(k))
    h = h - 1
    if h > k:
        raise Exception("Such tree does not exist")

    node = h + 1
    paths = [list(range(0, h + 1))]

    length_to_paths = defaultdict(list)

    for i in range(1, len(paths[0]) + 1):
        length_to_paths[i].append(paths[0])

    while node < k:
        ind = randint(1, h)

        path = choice(length_to_paths[ind])

        # ind = randint(1, min(len(path), h - 1))
        new_path = path[:ind] + [node]
        paths.append(new_path)

        for i in range(1, len(new_path) + 1):
            length_to_paths[i].append(new_path)
        node += 1

    tree = nx.prefix_tree(paths)
    tree.remove_node(-1)
    tree.remove_node(0)
    tree = nx.relabel_nodes(tree, {i: i - 1 for i in range(1, k + 1)})
    tree = tree.to_undirected()

    for i in range(k):
        tree.nodes[i]["nodes"] = [i]

    return tree


def calculate_subtree_size(tree: nx.Graph):
    subtree_sizes = dict(zip(list(tree.nodes), [1] * len(tree.nodes)))
    edges = list(nx.dfs_tree(tree).edges())

    for start, end in edges[::-1]:
        subtree_sizes[start] += subtree_sizes[end]

    return subtree_sizes


def calculate_all_ancestors(tree: nx.Graph):
    ancestors = dict(zip(list(tree.nodes), [[] for _ in range(len(tree.nodes))]))
    edges = list(nx.dfs_tree(tree).edges())

    for start, end in edges[::-1]:
        ancestors[start].extend([end] + ancestors[end])

    return ancestors


def partition(tree: nx.Graph):
    if not nx.is_tree(tree):
        raise Exception("Graph should be a tree")
    height = max(nx.shortest_path_length(tree, source=0).values()) + 1

    children = edges_union(list(nx.bfs_tree(tree, source=0).edges()))
    w = calculate_subtree_size(tree)
    ancestors = calculate_all_ancestors(tree)
    partitioned = set()

    used_names = defaultdict(int)
    components = {}
    parents = defaultdict(list)
    super_parents = {}
    all_nodes = set(tree.nodes())
    # tree = nx.dfs_tree(tree)

    solver = pywrapknapsack_solver.KnapsackSolver(
        pywrapknapsack_solver.KnapsackSolver.KNAPSACK_DYNAMIC_PROGRAMMING_SOLVER, '')

    comp = 1

    def partition_(node):
        nonlocal comp
        for child in children[node]:
            if w[child] >= height:
                partition_(child)

        w[node] = sum([w[child] for child in children[node]]) + 1

        while w[node] > height:
            part = []
            w_part = 0

            ws = [w[child] for child in children[node]]
            solver.Init(ws, [ws], [height])
            solver.Solve()

            for i in range(len(children[node])):
                if solver.BestSolutionContains(i):
                    child = children[node][i]
                    if w[child] == 0:
                        continue
                    w_part += w[child]
                    part.append(child)
            if w_part < height / 2:
                continue

            nodes = part + [n for node_ in part for n in ancestors[node_] if n not in partitioned]
            for node_ in nodes:
                partitioned.add(node_)
                for child in parents[node_]:
                    super_parents[child] = comp

            # partitioned.add(node)

            components[comp] = nodes
            parents[node].append(comp)
            used_names[node] += 1
            comp += 1

            w[node] -= w_part

            for node_ in part:
                w[node_] = 0

    partition_(0)

    last = list(all_nodes - partitioned)

    comp = 0
    for node_ in last:
        partitioned.add(node_)
        for child in parents[node_]:
            super_parents[child] = comp

    components[comp] = last
    graph = nx.Graph({k: {v: {}} for k, v in super_parents.items()})

    for i in range(len(components)):
        graph.nodes[i]["nodes"] = components[i]

    print("parents:", parents)
    print("ancestors:", ancestors)
    print("super parents:", super_parents)
    return graph


def generate_nodes(n: int, k: int, h: int = None, random_gen=np.random.rand, split_deviation=0.0, graph_partition=True):
    tree = generate_tree(k, h)
    # tree = nx.balanced_tree(math.ceil(k ** (1 / h)), h)
    height = max(nx.shortest_path_length(tree, source=0).values()) + 1

    if graph_partition:
        tree = partition(tree)

    print(nx.to_dict_of_dicts(tree))

    data = random_gen(n)

    split_ind = np.linspace(0, n, k + 1)[1: -1] + np.random.normal(0.0, split_deviation)
    split_ind = split_ind.astype(int)

    data_split = np.split(data, split_ind)

    bfs = list(nx.bfs_tree(tree, source=0).edges())

    network = Network()
    root = SuperNode(network, [data_split[node] for node in tree.nodes[0]["nodes"]])
    nodes = {0: root}

    for start, end in bfs[:k - 1]:
        if end == -1:
            continue
        nodes[end] = nodes[start].add_child([data_split[node] for node in tree.nodes[end]["nodes"]])

    return network, root, data, height
