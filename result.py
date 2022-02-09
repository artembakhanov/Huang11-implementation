import numpy as np


class Result:
    def __init__(self, sampled_values: np.ndarray, ranks: np.ndarray):
        self.sampled_values = sampled_values
        self.ranks = ranks
        self.value_to_rank = dict(zip(sampled_values, ranks))


class Summary:
    def __init__(self, sampled_values: np.ndarray, ranks: np.ndarray, ns: int, cls: int):
        self.sampled_values = sampled_values
        self.ranks = ranks
        self.value_to_rank = dict(zip(sampled_values, ranks))
        self.ns = ns
        self.cls = cls

    @property
    def large(self):
        return self.cls >= 0

    @staticmethod
    def merge(nk, *args):
        common_ns = sum([summary.ns for summary in args])

        if common_ns >= nk:
            # small-merge algorithm
            ...

        return ...
