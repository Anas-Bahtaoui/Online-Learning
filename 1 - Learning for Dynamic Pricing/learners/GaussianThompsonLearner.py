from typing import List, NamedTuple
import numpy as np
from BanditLearner import BanditLearner, BanditConfiguration
from entities import Product


class EstimationParameter(NamedTuple):
    t_0: float = 0.0001  # Precision
    mu_0: float = 1  # Mean


class GaussianTSLearner(BanditLearner):
    def __init__(self, config: BanditConfiguration):
        super().__init__(config)
        self.Q = [0 for _ in self._products]  # estimated reward
        self.parameters = [[EstimationParameter() for _ in product.candidate_prices] for product in
                           self._products]  # precision

    def _select_price_criteria(self, product: Product) -> List[float]:
        return [(np.random.randn() / (np.sqrt(self.parameters[product.id][i].t_0)) +
                 self.parameters[product.id][i].mu_0) for i in
                range(len(product.candidate_prices))]

    def _update(self):
        _, selected_price_indexes, product_rewards = self._experiment_history[-1]
        ### TODO: Why aren't we using customer count here?
        for p_i, reward in enumerate(product_rewards):
            self.Q[p_i] = (1 - 1.0 / self._environment.day) * self.Q[p_i] + (1.0 / self._environment.day) * reward
            i = selected_price_indexes[p_i]
            old_t, old_mu = self.parameters[p_i][i]
            new_mu = ((old_t * old_mu) + (self._environment.day * self.Q[p_i])) / (
                    old_t + self._environment.day)
            new_t = old_t + 1
            self.parameters[p_i][i] = EstimationParameter(t_0=new_t, mu_0=new_mu)

    def reset(self):
        self.__init__(self.config)
