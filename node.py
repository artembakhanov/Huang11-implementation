import enum
import math
from collections import defaultdict

import numpy as np
from typing import List, Dict
from abc import ABC, abstractmethod

from helper import ranks, sample
from message import Message, MessageType, NewTaskMessage, ResultMessage, NewNodeMessage, SuperNodeTaskResultMessage, \
    AggregateResultMessage, NewAggregateMessage, NewSuperNodeTaskMessage
from network import Network
from result import Result_, Result, Summary, SlaveResult
from task import Task
import sortednp as snp


class BasicNode:
    def recv_message(self, message, from_: ''):
        raise NotImplemented


class SlaveNode(BasicNode):
    def __init__(self, network: Network, data: np.ndarray, parent: 'SlaveNode'):
        self._network = network
        self._data = np.sort(data)
        self._local_ranks = ranks(self._data)
        self._children: List['SlaveNode'] = []
        self._parent = parent

        self._task = None
        self._aggregated_ranks = None
        self._merged = None

    def recv_message(self, message, from_: 'SlaveNode'):
        if message.type == MessageType.NEW_SUPER_NODE_TASK:
            new_message = SuperNodeTaskResultMessage(sample(self._data, self._local_ranks, message.data)[0])

        elif message.type == MessageType.NEW_SUPER_NODE_AGGREGATE:
            new_message = AggregateResultMessage(self.estimate_ranks(message.data))

        elif message.type == MessageType.SUPER_NODE_TASK_RESULT:
            self._task.add_results(from_, SlaveResult(message.data))
            if self._task.complete(len(self._children)):
                merged = self._task.results[0].data
                for data in self._task.results[1:]:
                    merged = snp.merge(merged, data.data, duplicates=snp.DROP)

                self._merged = merged

                new_message = NewAggregateMessage(merged)
                self._aggregated_ranks = self.estimate_ranks(merged)
                self._task = Task(0.0, 0.0)
                self._task.add_results(self, SlaveResult(None))

                for child in self._children:
                    self._network.send_message(new_message, self, child)

            return
        elif message.type == MessageType.SUPER_NODE_AGGREGATE_RESULT:
            assert self._aggregated_ranks is not None

            self._task.add_results(from_, SlaveResult(None))
            self._aggregated_ranks += message.data

            return
        else:
            raise Exception("Unknown message")

        self._network.send_message(new_message, self, self._parent)

    def estimate_ranks(self, x: np.ndarray):
        is_ = np.searchsorted(self._data, x, side='right')

        return is_

    def calculate_local_result(self, p: float):
        if not self.root:
            raise Exception("Non root slave node cannot aggregate data")

        self._task = Task(p, 0.0)
        self._aggregated_ranks = None
        self._merged = None

        message = NewSuperNodeTaskMessage(p)

        s, r = sample(self._data, self._local_ranks, p)
        self._task.add_results(self, SlaveResult(s))

        for child in self._children:
            self._network.send_message(message, self, child)

        if len(self._children) == 0:
            self._aggregated_ranks = r
            self._merged = s

        return self._merged, self._aggregated_ranks

    def add_child(self, data: np.ndarray):
        self._children.append(SlaveNode(self._network, data, self))

    @property
    def ns(self):
        return len(self._data)

    @property
    def root(self):
        return self._parent is None


class SlaveNodes:
    def __init__(self, network: Network, data: List[np.ndarray]):
        self._root = SlaveNode(network, data[0], None)
        self._ns = sum([len(data_) for data_ in data])

        for data_ in data[1:]:
            self.add_node(data_)

    def calculate_local_result(self, p: float):
        return self._root.calculate_local_result(p)

    def add_node(self, data: np.ndarray):
        self._ns += len(data)
        self._root.add_child(data)

    @property
    def ns(self):
        return self._ns


class SuperNode(BasicNode):
    def __init__(self, network: Network, data: List[np.ndarray], parent: 'SuperNode' = None, k=1, n=None,
                 slaves_nodes: 'SlaveNodes' = None):
        self._k = k
        self._n = n if n is not None else sum([len(data_) for data_ in data])
        self._parent = parent

        self._network = network
        self._children: List['SuperNode'] = []
        if slaves_nodes is None:
            self._slaves: 'SlaveNodes' = SlaveNodes(network, data)
        else:
            self._slaves = slaves_nodes
        self._task: Task = None
        self._local_result = None
        self._global_ranks = None
        self._value_to_rank = None

    @property
    def root(self):
        return self._parent is None

    @property
    def has_task(self):
        return self._task is not None

    @property
    def leaf(self):
        return len(self._children) == 0

    @property
    def sk(self):
        return math.sqrt(self._k)

    @property
    def nk(self):
        return self._n / math.sqrt(self._k)

    @property
    def large(self):
        return self.ns >= self._n / self.sk

    @property
    def ns(self):
        return self._slaves.ns

    def recv_message(self, message: Message, from_: 'SuperNode'):
        if message.type == MessageType.NEW_TASK:
            self.new_task(message.data)
        elif message.type == MessageType.RESULT:
            self.process_result(from_, message)

        elif message.type == MessageType.NEW_NODE:
            self._k += 1
            self._n += message.data

            self._resend_new_node_message(from_, message)
        else:
            raise Exception("Incorrect message")

    def add_child(self, data: List[np.ndarray]) -> 'SuperNode':
        new_n = sum([len(data_) for data_ in data])
        self._k += 1
        self._n += new_n

        node = SuperNode(self._network, data, self, self._k, self._n)
        self._resend_new_node_message(self, NewNodeMessage(new_n))

        self._children.append(node)

        return node

    def _resend_new_node_message(self, from_: 'SuperNode', message: Message):
        assert message.type == MessageType.NEW_NODE
        for child in self._children:
            if child != from_:
                self._network.send_message(message, self, child)

        if not self.root and self._parent != from_:
            self._network.send_message(message, self, self._parent)

    def new_task(self, eps: float):
        p = 1 / eps / self.ns if self.large else self.sk / eps / self._n
        message = NewTaskMessage(eps)
        self._task = Task(p, eps)

        self._local_result = self._calculate_local_result(eps)
        self._task.add_results(self, self._local_result)

        if self.leaf and not self.root:
            # it means we need to calculate just our values
            self._network.send_message(ResultMessage(self._local_result), self, self._parent)
            self._task = None

        else:
            for child in self._children:
                self._network.send_message(message, self, child)

    def _calculate_local_result(self, eps: float):
        return Result(self.nk, eps, self.sk,
                      Summary(self.ns, self.nk, eps, *self._slaves.calculate_local_result(self._task.p)))

    def process_result(self, from_: 'SuperNode', message: Message):
        self._task.add_results(from_, message.data)

        if self._task.complete(len(self._children)):
            if self.root:
                # here we run the merge algorithm and send the result message to the parent
                self._global_ranks, self._value_to_rank = self._task.global_ranks()
            else:
                result = self._task.merge_results()
                self._network.send_message(ResultMessage(result), self, self._parent)

    def rank_query(self, r: float, return_rank=False):
        if self._global_ranks is None:
            return 0

        ind = np.argmin((self._global_ranks - r) ** 2)
        if return_rank:
            return self._value_to_rank[ind], self._global_ranks[ind]
        else:
            return self._value_to_rank[ind]

    def __repr__(self):
        return f"Node(network={self._network}, parent={id(self._parent)})"
