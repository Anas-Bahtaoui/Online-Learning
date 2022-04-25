from dataclasses import dataclass
from typing import TypeVar, List

import numpy as np

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

    def get_expectation(self) -> float:
        return self.mean

    def get_sample_value(self) -> float:
        return np.random.normal(self.mean, self.variance)


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
        return self.alpha

    def get_sample_value(self) -> List[float]:
        return list(np.random.dirichlet(np.array(self.alpha)))

