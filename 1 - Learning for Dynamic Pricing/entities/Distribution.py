from dataclasses import dataclass
from functools import lru_cache
from typing import TypeVar, List

import scipy

from random_ import np_random
import numpy as np
from scipy.stats import dirichlet

T = TypeVar("T")


class AbstractDistribution:
    def get_expectation(self) -> T:
        raise NotImplementedError()

    def get_sample_value(self) -> T:
        raise NotImplementedError()


@dataclass
class NormalGaussian(AbstractDistribution):
    mean: float
    variance: float

    def __post_init__(self):
        if self.variance <= 0.00001:
            raise ValueError("Variance must be strictly positive")

    def get_expectation(self) -> float:
        return self.mean

    def get_sample_value(self) -> float:
        return np_random.normal(self.mean, self.variance)

    @lru_cache(maxsize=None)
    def calculate_ratio_of(self, value: float) -> float:
        """
       This function calculates a value based on the inverse of the normal distribution function.
       So, it calculates the ratio of the people who have the reservation price beneath their expectation.
       We only need this in Greedy and in clairvoyant because we basically expect this to be something similar to emulating each customer.
       :param value: The suggested price of the product
       :return: ratio of people who have the reservation price beneath their expectation
       """
        return 1 - scipy.stats.norm.cdf((value - self.mean) / self.variance)

    def __hash__(self):
        return hash((self.mean, self.variance))

class PositiveIntegerGaussian(NormalGaussian):
    def get_sample_value(self) -> int:
        result = -1
        while result <= 0:
            result = round(super().get_sample_value())
        return result

    def get_expectation(self) -> int:
        return round(super().get_expectation())


@dataclass
class Dirichlet(AbstractDistribution):
    alpha: List[float]

    def get_expectation(self) -> List[float]:
        return dirichlet.mean(self.alpha)

    def get_sample_value(self) -> List[float]:
        return list(np_random.dirichlet(np.array(self.alpha)))


@dataclass
class Constant(AbstractDistribution):
    def get_sample_value(self) -> float:
        return self.value

    value: float

    def get_expectation(self) -> float:
        return self.value


@dataclass
class Poisson(AbstractDistribution):
    mean: float

    def get_sample_value(self) -> int:
        return np_random.poisson(self.mean)

    def get_expectation(self) -> int:
        return round(self.mean)
