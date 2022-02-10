from typing import Dict

import numpy as np

from result import Result_, Result


class Task:
    def __init__(self, p: float, eps: float):
        self._p = p
        self._eps = eps
        self._results: Dict['Node', 'Result'] = {}

    def __repr__(self) -> str:
        return f"Task(eps={self._eps}, p={self.p})"

    @property
    def p(self):
        return self._p

    @property
    def ip(self):
        return 1 / self.p

    @property
    def eps(self):
        return self._eps

    def node_result(self, node: 'Node'):
        if node not in self._results:
            raise Exception("No results exist for this node")

        return self._results[node]

    def r(self, a: float, node: 'Node'):

        node_result = self.node_result(node)

        if a not in node_result.value_to_rank:
            raise Exception("No such value exist for this node")

        return node_result.value_to_rank[a]

    def pred(self, x: float, node: 'Node'):
        s = self.node_result(node).sampled_values

        return s[s <= x].max()

    def rpred(self, x: float, node: 'Node'):
        try:
            pred = self.pred(x, node)
        except ValueError:
            return -self.ip

        return self.node_result(node).value_to_rank[pred]

    def rank(self, a: float, node: 'Node'):
        try:
            return self.r(a, node)
        except:
            return self.rpred(a, node) + self.ip

    def global_rank(self, a: float):
        res = 0
        for node in self._results.keys():
            res += self.rank(a, node)

        return res

    def global_ranks(self):
        return Result.merge_non_empty(list(self._results.values()), merge_summaries=False).global_ranks()

    def add_results(self, node: 'Node', results: 'Result'):
        self._results[node] = results

    def merge_results(self, merge_summaries=True):
        return Result.merge_non_empty(list(self._results.values()), merge_summaries=merge_summaries)

    def complete(self, n: int):
        return len(self._results) == n + 1
