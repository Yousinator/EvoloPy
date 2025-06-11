# -*- coding: utf-8 -*-
"""
Created on Tue May 17 12:46:20 2016

@author: Hossam Faris

updated on Sun Feb 9 06:05:50 2025
"""
import numpy as np
import math



class BaseBenchmark:
    """Base class for all benchmark functions"""
    def __init__(self):
        self.name = self.__class__.__name__
        self.lb = -100
        self.ub = 100
        self.dim = 30

    def evaluate(self, x):
        """Evaluate the benchmark function at point x"""
        raise NotImplementedError("Subclasses must implement evaluate()")

    def prod(self, it):
        p = 1
        for n in it:
            p *= n
        return p


    def Ufun(self, x, a, k, m):
        y = k * ((x - a) ** m) * (x > a) + k * ((-x - a) ** m) * (x < (-a))
        return y


class F1(BaseBenchmark):
    """Sphere function - Unimodal, separable"""
    def __init__(self):
        super().__init__()
        self.lb = -100
        self.ub = 100
        self.dim = 30

    def evaluate(self, x):
        return np.sum(x ** 2)

class F3(BaseBenchmark):
    """Step function - Unimodal, non-separable"""
    def __init__(self):
        super().__init__()
        self.lb = -100
        self.ub = 100
        self.dim = 30

    def evaluate(self, x):
        dim = len(x) + 1
        o = 0
        for i in range(1, dim):
            o = o + (np.sum(x[0:i])) ** 2
        return o

class F4(BaseBenchmark):
    """Schwefel 2.21 function - Unimodal, non-separable"""
    def __init__(self):
        super().__init__()
        self.lb = -100
        self.ub = 100
        self.dim = 30

    def evaluate(self, x):
        return max(abs(x))

class F5(BaseBenchmark):
    """Rosenbrock function - Unimodal, non-separable"""
    def __init__(self):
        super().__init__()
        self.lb = -30
        self.ub = 30
        self.dim = 30

    def evaluate(self, x):
        dim = len(x)
        return np.sum(100 * (x[1:dim] - (x[0:dim-1] ** 2)) ** 2 + (x[0:dim-1] - 1) ** 2)

class F6(BaseBenchmark):
    """Step function - Unimodal, separable"""
    def __init__(self):
        super().__init__()
        self.lb = -100
        self.ub = 100
        self.dim = 30

    def evaluate(self, x):
        return np.sum(abs((x + 0.5)) ** 2)

class F7(BaseBenchmark):
    """Quartic function with noise - Unimodal, separable"""
    def __init__(self):
        super().__init__()
        self.lb = -1.28
        self.ub = 1.28
        self.dim = 30

    def evaluate(self, x):
        dim = len(x)
        w = np.arange(1, dim + 1)
        return np.sum(w * (x ** 4)) + np.random.uniform(0, 1)

class F8(BaseBenchmark):
    """Schwefel function - Multimodal, separable"""
    def __init__(self):
        super().__init__()
        self.lb = -500
        self.ub = 500
        self.dim = 30

    def evaluate(self, x):
        return sum(-x * (np.sin(np.sqrt(abs(x)))))

class F9(BaseBenchmark):
    """Rastrigin function - Multimodal, separable"""
    def __init__(self):
        super().__init__()
        self.lb = -5.12
        self.ub = 5.12
        self.dim = 30

    def evaluate(self, x):
        dim = len(x)
        return np.sum(x ** 2 - 10 * np.cos(2 * math.pi * x)) + 10 * dim

class F10(BaseBenchmark):
    """Ackley function - Multimodal, non-separable"""
    def __init__(self):
        super().__init__()
        self.lb = -32
        self.ub = 32
        self.dim = 30

    def evaluate(self, x):
        dim = len(x)
        return (-20 * np.exp(-0.2 * np.sqrt(np.sum(x ** 2) / dim))
                - np.exp(np.sum(np.cos(2 * math.pi * x)) / dim)
                + 20 + np.exp(1))

class F11(BaseBenchmark):
    """Griewank function - Multimodal, non-separable"""
    def __init__(self):
        super().__init__()
        self.lb = -600
        self.ub = 600
        self.dim = 30

    def evaluate(self, x):
        dim = len(x)
        w = np.arange(1, dim + 1)
        return np.sum(x ** 2) / 4000 - np.prod(np.cos(x / np.sqrt(w))) + 1

class F12(BaseBenchmark):
    """Levy function - Multimodal, non-separable"""
    def __init__(self):
        super().__init__()
        self.lb = -50
        self.ub = 50
        self.dim = 30

    def evaluate(self, x):
        dim = len(x)
        return (math.pi / dim) * (
            10 * ((np.sin(math.pi * (1 + (x[0] + 1) / 4))) ** 2)
            + np.sum(
                (((x[:dim-1] + 1) / 4) ** 2)
                * (1 + 10 * ((np.sin(math.pi * (1 + (x[1:] + 1) / 4)))) ** 2)
            )
            + ((x[dim-1] + 1) / 4) ** 2
        ) + np.sum(self.Ufun(x, 10, 100, 4))

class F13(BaseBenchmark):
    """Levy 13 function - Multimodal, non-separable"""
    def __init__(self):
        super().__init__()
        self.lb = -50
        self.ub = 50
        self.dim = 30

    def evaluate(self, x):
        if x.ndim == 1:
            x = x.reshape(1, -1)
        return 0.1 * (
            (np.sin(3 * np.pi * x[:,0])) ** 2
            + np.sum(
                (x[:,:-1] - 1) ** 2
                * (1 + (np.sin(3 * np.pi * x[:,1:])) ** 2), axis=1
            )
            + ((x[:,-1] - 1) ** 2) * (1 + (np.sin(2 * np.pi * x[:,-1])) ** 2)
        ) + np.sum(self.Ufun(x, 5, 100, 4))

class F14(BaseBenchmark):
    """Shekel's Foxholes function - Multimodal, non-separable"""
    def __init__(self):
        super().__init__()
        self.lb = -65.536
        self.ub = 65.536
        self.dim = 2

    def evaluate(self, x):
        aS = [
            [-32, -16, 0, 16, 32, -32, -16, 0, 16, 32, -32, -16, 0, 16, 32,
             -32, -16, 0, 16, 32, -32, -16, 0, 16, 32],
            [-32, -32, -32, -32, -32, -16, -16, -16, -16, -16, 0, 0, 0, 0, 0,
             16, 16, 16, 16, 16, 32, 32, 32, 32, 32]
        ]
        aS = np.asarray(aS)
        bS = np.zeros(25)
        v = np.array(x)
        for i in range(25):
            H = v - aS[:, i]
            bS[i] = np.sum((np.power(H, 6)))
        w = np.arange(1, 26)
        return ((1.0 / 500) + np.sum(1.0 / (w + bS))) ** (-1)

class F16(BaseBenchmark):
    """Six-hump camel back function - Multimodal, non-separable"""
    def __init__(self):
        super().__init__()
        self.lb = -5
        self.ub = 5
        self.dim = 2

    def evaluate(self, x):
        return (4 * (x[0] ** 2) - 2.1 * (x[0] ** 4) + (x[0] ** 6) / 3
                + x[0] * x[1] - 4 * (x[1] ** 2) + 4 * (x[1] ** 4))

class F17(BaseBenchmark):
    """Branin function - Multimodal, non-separable"""
    def __init__(self):
        super().__init__()
        self.lb = -5
        self.ub = 15
        self.dim = 2

    def evaluate(self, x):
        return ((x[1] - (x[0] ** 2) * 5.1 / (4 * (np.pi ** 2)) + 5 / np.pi * x[0] - 6) ** 2
                + 10 * (1 - 1 / (8 * np.pi)) * np.cos(x[0]) + 10)

class F18(BaseBenchmark):
    """Goldstein-Price function - Multimodal, non-separable"""
    def __init__(self):
        super().__init__()
        self.lb = -2
        self.ub = 2
        self.dim = 2

    def evaluate(self, x):
        return (1 + (x[0] + x[1] + 1) ** 2 * (19 - 14 * x[0] + 3 * (x[0] ** 2)
                - 14 * x[1] + 6 * x[0] * x[1] + 3 * x[1] ** 2)) * (
            30 + (2 * x[0] - 3 * x[1]) ** 2 * (18 - 32 * x[0] + 12 * (x[0] ** 2)
                + 48 * x[1] - 36 * x[0] * x[1] + 27 * (x[1] ** 2)))

class F20(BaseBenchmark):
    """Hartman 6 function - Multimodal, non-separable"""
    def __init__(self):
        super().__init__()
        self.lb = 0
        self.ub = 1
        self.dim = 6

    def evaluate(self, x):
        aH = [
            [10, 3, 17, 3.5, 1.7, 8],
            [0.05, 10, 17, 0.1, 8, 14],
            [3, 3.5, 1.7, 10, 17, 8],
            [17, 8, 0.05, 10, 0.1, 14]
        ]
        aH = np.asarray(aH)
        cH = [1, 1.2, 3, 3.2]
        cH = np.asarray(cH)
        pH = [
            [0.1312, 0.1696, 0.5569, 0.0124, 0.8283, 0.5886],
            [0.2329, 0.4135, 0.8307, 0.3736, 0.1004, 0.9991],
            [0.2348, 0.1415, 0.3522, 0.2883, 0.3047, 0.6650],
            [0.4047, 0.8828, 0.8732, 0.5743, 0.1091, 0.0381]
        ]
        pH = np.asarray(pH)
        o = 0
        for i in range(4):
            o = o - cH[i] * np.exp(-(np.sum(aH[i, :] * ((x - pH[i, :]) ** 2))))
        return o

class F21(BaseBenchmark):
    """Shekel 5 function - Multimodal, non-separable"""
    def __init__(self):
        super().__init__()
        self.lb = 0
        self.ub = 10
        self.dim = 4

    def evaluate(self, x):
        aSH = [
            [4, 4, 4, 4],
            [1, 1, 1, 1],
            [8, 8, 8, 8],
            [6, 6, 6, 6],
            [3, 7, 3, 7]
        ]
        cSH = [0.1, 0.2, 0.2, 0.4, 0.4]
        aSH = np.asarray(aSH)
        cSH = np.asarray(cSH)
        fit = 0
        for i in range(5):
            v = np.array(x - aSH[i, :])
            fit = fit - ((v) * (v.T) + cSH[i]) ** (-1)
        return fit.item(0)

class F22(BaseBenchmark):
    """Shekel 7 function - Multimodal, non-separable"""
    def __init__(self):
        super().__init__()
        self.lb = 0
        self.ub = 10
        self.dim = 4

    def evaluate(self, x):
        aSH = [
            [4, 4, 4, 4],
            [1, 1, 1, 1],
            [8, 8, 8, 8],
            [6, 6, 6, 6],
            [3, 7, 3, 7],
            [2, 9, 2, 9],
            [5, 5, 3, 3]
        ]
        cSH = [0.1, 0.2, 0.2, 0.4, 0.4, 0.6, 0.3]
        aSH = np.asarray(aSH)
        cSH = np.asarray(cSH)
        fit = 0
        for i in range(7):
            v = np.array(x - aSH[i, :])
            fit = fit - ((v) * (v.T) + cSH[i]) ** (-1)
        return fit.item(0)

class F23(BaseBenchmark):
    """Shekel 10 function - Multimodal, non-separable"""
    def __init__(self):
        super().__init__()
        self.lb = 0
        self.ub = 10
        self.dim = 4

    def evaluate(self, x):
        aSH = [
            [4, 4, 4, 4],
            [1, 1, 1, 1],
            [8, 8, 8, 8],
            [6, 6, 6, 6],
            [3, 7, 3, 7],
            [2, 9, 2, 9],
            [5, 5, 3, 3],
            [8, 1, 8, 1],
            [6, 2, 6, 2],
            [7, 3.6, 7, 3.6]
        ]
        cSH = [0.1, 0.2, 0.2, 0.4, 0.4, 0.6, 0.3, 0.7, 0.5, 0.5]
        aSH = np.asarray(aSH)
        cSH = np.asarray(cSH)
        fit = 0
        for i in range(10):
            v = np.array(x - aSH[i, :])
            fit = fit - ((v) * (v.T) + cSH[i]) ** (-1)
        return fit.item(0)

class Ackley(BaseBenchmark):
    """Ackley function - Multimodal, non-separable"""
    def __init__(self):
        super().__init__()
        self.lb = -32.768
        self.ub = 32.768
        self.dim = 30

    def evaluate(self, x):
        a = 20
        b = 0.2
        c = 2 * np.pi
        d = len(x)
        sum1 = np.sum(x**2)
        sum2 = np.sum(np.cos(c * x))
        return -a * np.exp(-b * np.sqrt(sum1 / d)) - np.exp(sum2 / d) + a + np.e

class Rosenbrock(BaseBenchmark):
    """Rosenbrock function - Unimodal, non-separable"""
    def __init__(self):
        super().__init__()
        self.lb = -5
        self.ub = 10
        self.dim = 30

    def evaluate(self, x):
        return sum(100.0 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)

class Rastrigin(BaseBenchmark):
    """Rastrigin function - Multimodal, separable"""
    def __init__(self):
        super().__init__()
        self.lb = -5.12
        self.ub = 5.12
        self.dim = 30

    def evaluate(self, x):
        A = 10
        return A * len(x) + np.sum(x**2 - A * np.cos(2 * np.pi * x))

class Griewank(BaseBenchmark):
    """Griewank function - Multimodal, non-separable"""
    def __init__(self):
        super().__init__()
        self.lb = -600
        self.ub = 600
        self.dim = 30

    def evaluate(self, x):
        part1 = np.sum(x**2) / 4000
        part2 = np.prod(np.cos(x / np.sqrt(np.arange(1, len(x) + 1))))
        return part1 - part2 + 1

class CustomObjFunction(BaseBenchmark):
    """Class for custom objective functions"""
    def __init__(self, func, name="Custom", lb=-100, ub=100, dim=30):
        """
        Initialize a custom objective function

        Parameters
        ----------
        func : callable
            The objective function to be minimized
        name : str, optional
            Name of the function
        lb : float or list, optional
            Lower bounds for the search space
        ub : float or list, optional
            Upper bounds for the search space
        dim : int, optional
            Number of dimensions in the search space
        """
        super().__init__()
        self.name = name
        self.lb = lb
        self.ub = ub
        self.dim = dim
        self._func = func

    def evaluate(self, x):
        return self._func(x)


