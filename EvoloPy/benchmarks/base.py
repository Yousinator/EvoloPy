import numpy as np

class BaseBenchmark:
    """Base class for all benchmark functions"""

    def __init__(self, lb, ub, dim):
        self.lb = lb
        self.ub = ub
        self.dim = dim

    def evaluate(self, x):
        """Evaluate the benchmark function at point x"""
        raise NotImplementedError("Subclasses must implement evaluate()")

    def prod(self, it):
        """Calculate product of iterable"""
        p = 1
        for n in it:
            p *= n
        return p

    def Ufun(self, x, a, k, m):
        """Utility function used in some benchmarks"""
        y = k * ((x - a) ** m) * (x > a) + k * ((-x - a) ** m) * (x < (-a))
        return y