import enum
import math

import numpy as np
from typing import List, Dict
from abc import ABC, abstractmethod

from helper import ranks, sample
from message import Message, MessageType, NewTaskMessage, ResultMessage, NewNodeMessage
from network import Network
from result import Result
from task import Task


class Node:
    def __init__(self, network: Network, data: np.ndarray, parent: 'Node' = None, k=1, n=None):
        self._k = k
        self._n = n if n is not None else len(data)
        self._parent = parent
        self._data = np.sort(data)
        self._local_ranks = ranks(self._data)

        self._network = network
        self._children: List['Node'] = []
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
    def large(self):
        return len(self._data) >= self._n / self.sk

    def recv_message(self, message: Message, from_: 'Node'):
        if message.type == MessageType.NEW_TASK:
            self.new_task(message.data['alpha'], message.data['epsilon'])
        elif message.type == MessageType.RESULT:
            self.process_result(from_, message)

        elif message.type == MessageType.NEW_NODE:
            self._k += 1
            self._n += message.data

            self._resend_new_node_message(from_, message)
        else:
            raise Exception("Incorrect message")

    def add_child(self, data: np.ndarray) -> 'Node':
        self._k += 1
        self._n += len(data)

        node = Node(self._network, data, self, self._k, self._n)
        self._resend_new_node_message(self, NewNodeMessage(len(data)))

        self._children.append(node)

        return node

    def _resend_new_node_message(self, from_: 'Node', message: Message):
        assert message.type == MessageType.NEW_NODE
        for child in self._children:
            if child != from_:
                self._network.send_message(message, self, child)

        if not self.root and self._parent != from_:
            self._network.send_message(message, self, self._parent)

    def new_task(self, alpha: float, epsilon: float):
        p = 1 / epsilon / len(self._data) if self.large else self.sk / epsilon / self._n
        message = NewTaskMessage(alpha, epsilon)
        self._task = Task(alpha, p)

        sampled_data = sample(self._data, self._local_ranks, p)
        self._local_result = Result(*sampled_data)
        self._task.add_results(self, self._local_result)

        if self.leaf and not self.root:
            # it means we need to calculate just our values
            self._network.send_message(ResultMessage(self._local_result), self, self._parent)
            self._task = None

        else:
            for child in self._children:
                self._network.send_message(message, self, child)

    def process_result(self, from_: 'Node', message: Message):
        self._task.add_results(from_, message.data)

        if self._task.complete(len(self._children)):
            # here we run the merge algorithm and send the result message to the parent
            self._global_ranks, self._value_to_rank = self._task.global_ranks()

    def rank_query(self, r: float):
        if self._global_ranks is None:
            return 0

        return self._value_to_rank[np.argmin((self._global_ranks - r) ** 2)]

    def r(self, a: float, node: 'Node' = None):
        if not self.has_task:
            return 0

        if node is None:
            node = self

        return self._task.r(a, node)

    def pred(self, x, node: 'Node' = None):
        if not self.has_task:
            return 0

        if node is None:
            node = self

        return self._task.pred(x, node)

    def rpred(self, x, node: 'Node' = None):
        if not self.has_task:
            return 0

        if node is None:
            node = self

        return self._task.rpred(x, node)

    def global_rank(self, a):
        if not self.has_task:
            return 0

        return self._task.global_rank(a)

    def r_hat_node(self, x, i=-1):
        if not self.has_task:
            return 0

        return self.rpred(x) + self._task.ip

    def r_hat(self, a, i):
        if not self.has_task:
            return 0

    def __repr__(self):
        return f"Node(network={self._network}, data={self._data}, parent={id(self._parent)})"
