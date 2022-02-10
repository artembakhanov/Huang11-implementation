import math
from collections import defaultdict
from typing import List

import numpy as np

from helper import sample


class Result_:
    def __init__(self, sampled_values: np.ndarray, ranks: np.ndarray):
        self.sampled_values = sampled_values
        self.ranks = ranks
        self.value_to_rank = dict(zip(sampled_values, ranks))


class Result:
    def __init__(self, nk: float, eps: float, sk: float, *summaries: 'Summary'):
        self._nk = nk
        self._eps = eps
        self._sk = sk
        self.summaries = defaultdict(list)
        for summary in summaries:
            self.summaries[summary.cls].append(summary)

    @classmethod
    def merge(cls, nk: float, eps: float, sk: float, results: List['Result'], merge_summaries: bool = True):
        res = Result(nk, eps, sk)
        for result in results:
            for cls_, summaries in result.summaries.items():
                res.summaries[cls_].extend(summaries)

        if merge_summaries:
            res._merge()

        return res

    @classmethod
    def merge_non_empty(cls, results: List['Result'], merge_summaries: bool = True):
        if len(results) == 0:
            raise Exception("Should have at least one result")

        result = results[0]
        nk = result._nk
        eps = result._eps
        sk = result._sk

        return Result.merge(nk, eps, sk, results, merge_summaries=merge_summaries)

    def _merge(self):
        for cls in range(-1, math.ceil(math.log2(self._sk))):
            summaries = self.summaries[cls]
            if cls == -1:
                new_summaries, new_cls = Summary.merge_small(self._nk, *summaries)
                self.summaries[cls] = new_summaries
                continue

            while len(summaries) > 1:
                sum1, sum2 = summaries.pop(), summaries.pop()
                new_summaries, new_cls = sum1.merge_large()
                self.summaries[new_cls].extend(new_summaries)

    def rank(self, a: float):
        res = 0
        for list_sum in self.summaries.values():
            for sum_ in list_sum:
                res += sum_.rank(a)

        return res

    def global_ranks(self):
        value_to_rank = {}
        ranks = []
        calc = set()
        calc_ = []
        for list_sum in self.summaries.values():
            for sum_ in list_sum:
                for a in sum_.sampled_values:
                    if a not in calc:
                        r = self.rank(a)
                        value_to_rank[len(ranks)] = a
                        ranks.append(r)
                        calc.add(a)
                        calc_.append(a)

        return np.array(ranks), value_to_rank

    def __len__(self):
        return sum(len(sum_) for sum_list in self.summaries.values() for sum_ in sum_list)


class Summary:
    def __init__(self, ns: int, nk: float, eps: float, sampled_values: np.ndarray, ranks: np.ndarray):
        self._sampled_values = sampled_values
        self._ranks = ranks
        self._value_to_rank = dict(zip(sampled_values, ranks))
        self._ns: int = ns
        self._nk: float = nk
        self._kn: float = 1 / self._nk
        self._eps: float = eps
        self._cls: int = math.floor(math.log2(self._ns * self._kn)) if ns > self._nk else -1
        self._ps: float = 1 / self._eps / self._ns if self.large else self._kn / self._eps

    @property
    def large(self):
        return self._cls >= 0

    @property
    def ips(self):
        return 1 / self._ps

    @property
    def ns(self):
        return self._ns

    @property
    def sampled_values(self):
        return self._sampled_values

    @property
    def ranks(self):
        return self._ranks

    @property
    def cls(self):
        return self._cls

    @staticmethod
    def merge_small(nk, *args: 'Summary'):
        common_ns = sum([summary.ns for summary in args])

        sampled_values = []
        ranks = []
        used = set()
        p = nk / common_ns

        if common_ns >= nk:
            # small-merge algorithm, return one summary
            for summary in args:
                s, _ = sample(summary.sampled_values, summary.ranks, p)
                for a in s:
                    if a not in used:
                        sampled_values.append(a)
                        rank = sum([sum_.rank(a) for sum_ in args])
                        ranks.append(rank)
                        used.add(a)

            new_summary = args[0].update(sampled_values, ranks, common_ns)

            return [new_summary], new_summary._cls

        return args, -1  # the same list of summaries

    def merge_large(self, o: 'Summary'):
        assert self._cls == o._cls

        common_ns = self._ns + o._ns

        p1 = self._ns / common_ns
        p2 = o._ns / common_ns

        summaries = zip([p1, p2], [self, o])

        sampled_values = []
        ranks = []
        used = set()

        for p, summary in summaries:
            s, _ = sample(summary.sampled_values, summary.ranks, p)

            for a in s:
                if a not in used:
                    sampled_values.append(a)
                    rank = sum([sum_[1].rank(a) for sum_ in summaries])
                    ranks.append(rank)
                    used.add(a)

        new_summary = self.update(sampled_values, ranks, common_ns)

        return [new_summary], new_summary._cls

    def update(self, new_sampled_values, new_ranks, new_ns):
        return Summary(new_sampled_values, new_ranks, new_ns, self._nk, self._eps)

    def r(self, a: float):
        if a not in self._value_to_rank:
            raise Exception("No such value exist for this node")

        return self._value_to_rank[a]

    def pred(self, x: float):
        s = self._sampled_values

        return s[s <= x].max()

    def rpred(self, x: float):
        try:
            pred = self.pred(x)
        except ValueError:
            return -self.ips

        return self._value_to_rank[pred]

    def rank(self, a: float):
        try:
            return self.r(a)
        except:
            return self.rpred(a) + self.ips

    def __len__(self):
        return len(self._sampled_values)
