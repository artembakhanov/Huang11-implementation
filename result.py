import math
from collections import defaultdict
from typing import List
import sortednp as snp

import numpy as np

from helper import sample
from hirola import HashTable


class Result_:
    def __init__(self, sampled_values: np.ndarray, ranks: np.ndarray):
        self.sampled_values = sampled_values
        self.ranks = ranks
        self.value_to_rank = dict(zip(sampled_values, ranks))


class BaseResult:
    pass


class SlaveResult(BaseResult):
    def __init__(self, data: np.ndarray):
        self._data = data

    @property
    def data(self):
        return self._data


class Result(BaseResult):
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
                if new_cls != -1:
                    self.summaries[cls] = []
                    self.summaries[new_cls].extend(new_summaries)

                continue

            while len(summaries) > 1:
                sum1, sum2 = summaries.pop(), summaries.pop()
                new_summaries, new_cls = sum1.merge_large(sum2)
                self.summaries[new_cls].extend(new_summaries)

    def rank(self, a: float):
        res = 0
        for list_sum in self.summaries.values():
            for sum_ in list_sum:
                res += sum_.rank(a)

        return res

    def rank_a(self, a: np.ndarray):
        res = np.zeros(a.shape, dtype=a.dtype)
        for list_sum in self.summaries.values():
            for sum_ in list_sum:
                res += sum_.rank_a(a)

        return res

    def global_ranks(self):
        value_to_rank = {}
        ranks = []
        calc = set()
        calc_ = []
        for list_sum in self.summaries.values():
            for sum_ in list_sum:
                s = sum_.sampled_values
                r = self.rank_a(s)
                for i in range(len(r)):
                    if s[i] not in calc:
                        value_to_rank[len(ranks)] = s[i]
                        ranks.append(r[i])
                        calc.add(s[i])

                # for a in sum_.sampled_values:
                #     if a not in calc:
                #         r = self.rank(a)
                #         value_to_rank[len(ranks)] = a
                #         ranks.append(r)
                #         calc.add(a)
                #         calc_.append(a)

        return np.array(ranks), value_to_rank

    def __len__(self):
        return sum(len(sum_) for sum_list in self.summaries.values() for sum_ in sum_list)


class Summary:
    def __init__(self, ns: int, nk: float, eps: float, sampled_values: np.ndarray, ranks: np.ndarray):
        self._sampled_values = np.append(sampled_values, [np.NaN])
        self._ranks = ranks
        # self._value_to_ind = HashTable(len(sampled_values) * 1.25, np.float)  # dict(zip(sampled_values, ranks))
        # self._rank_ind = self._value_to_ind.add(sampled_values)
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
        return self._sampled_values[:-1]

    @property
    def ranks(self):
        return self._ranks

    @property
    def cls(self):
        return self._cls

    @staticmethod
    def merge_small(nk, *args: 'Summary'):
        common_ns = sum([summary.ns for summary in args])
        if common_ns == 0:
            return [], -1

        sampled_values = np.zeros((0,))
        ranks = np.zeros((0,))
        used = set()
        p = nk / common_ns

        large = common_ns >= nk

        # small-merge algorithm, return one summary
        for summary in args:
            sampled_values_ = []
            ranks_ = []
            if large:
                s, _ = sample(summary.sampled_values, summary.ranks, p)
            else:
                s = summary.sampled_values

            r = np.zeros(s.shape)
            for sum_ in args:
                r += sum_.rank_a(s, sum_ == summary)

            for i in range(len(r)):
                a = s[i]
                rank = r[i]
                sampled_values_.append(a)
                ranks_.append(rank)
                used.add(a)

            sampled_values, ind = snp.merge(sampled_values, np.array(sampled_values_), indices=True)
            ranks_np = np.zeros(len(sampled_values))
            ranks_np[ind[0]] = ranks
            ranks_np[ind[1]] = ranks_
            ranks = ranks_np

            # for a in s:
            #     if a not in used:
            #         sampled_values.append(a)
            #         rank = sum([sum_.rank(a) for sum_ in args])
            #         r.append(rank)
            #         used.add(a)

        new_summary = args[0].update(np.array(sampled_values), np.array(ranks), common_ns)

        return [new_summary], new_summary._cls

    def merge_large(self, o: 'Summary'):
        assert self._cls == o._cls

        common_ns = self._ns + o._ns

        p1 = self._ns / common_ns
        p2 = o._ns / common_ns

        summaries = list(zip([p1, p2], [self, o]))

        sampled_values = np.zeros((0,))
        ranks = np.zeros((0,))
        used = set()

        for p, summary in summaries:
            s, _ = sample(summary.sampled_values, summary.ranks, p)

            sampled_values_ = []
            ranks_ = []
            r = np.zeros(s.shape)

            for sum_ in summaries:
                r += sum_[1].rank_a(s, summary == sum_)

            for i in range(len(r)):
                a = s[i]
                rank = r[i]
                sampled_values_.append(a)
                ranks_.append(rank)
                used.add(a)

            sampled_values, ind = snp.merge(sampled_values, np.array(sampled_values_), indices=True)
            ranks_np = np.zeros(len(sampled_values))
            ranks_np[ind[0]] = ranks
            ranks_np[ind[1]] = ranks_
            ranks = ranks_np
            # for a in s:
            #     if a not in used:
            #         sampled_values.append(a)
            #         rank = sum([sum_[1].rank(a) for sum_ in summaries])
            #         ranks.append(rank)
            #         used.add(a)

        new_summary = self.update(sampled_values, ranks, common_ns)

        return [new_summary], new_summary._cls

    def update(self, new_sampled_values, new_ranks, new_ns):
        return Summary(new_ns, self._nk, self._eps, np.array(new_sampled_values), np.array(new_ranks))

    def r(self, a: float):
        if a not in self._value_to_rank:
            return None

        return self._value_to_rank[a]

    def r_a(self, a: np.ndarray, other=np.NaN):
        return np.array([self._value_to_rank.get(aa, other) for aa in a])

    def pred(self, x: float):
        s = self._sampled_values[:-1]

        i = np.searchsorted(s, x, side='right') - 1

        return None if i == -1 else s[i]
        # v = s[s <= x]
        # return None if len(v) == 0 else v.max()

    def pred_a(self, x: np.ndarray):
        s = self._sampled_values
        is_ = np.searchsorted(s, x, side='right') - 1
        return s[is_]

    def rpred(self, x: float):
        pred = self.pred(x)
        if pred is None:
            return -self.ips

        return self._value_to_rank[pred]

    def rpred_a(self, x: np.ndarray, own=False):
        # pred = self.pred_a(x)
        #
        # return self.r_a(pred, -self.ips)
        s = self._sampled_values
        # ind_present = self._value_to_ind.contains(x)

        is_ = (np.searchsorted(s, x, side='right') - 1)
        ranks = self._ranks[is_].astype(float)
        ranks[is_ == -1] = 0.0 if own else -self.ips
        # ranks[ind_present] -= self.ips

        return ranks

    def rank(self, a: float):
        res = self.r(a)
        if res is None:
            res = self.rpred(a) + self.ips

        return res

    def rank_a(self, a: np.ndarray, own=False):
        # res = self.r_a(a)
        #
        # other = a[np.isnan(res)]
        # res[np.isnan(res)] = self.rpred_a(other) + self.ips
        if len(self.sampled_values) == 0:
            return 0.0
        res = self.rpred_a(a) + (0.0 if own else self.ips)
        return res

    def __len__(self):
        return len(self._sampled_values) - 1
