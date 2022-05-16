from typing import List, NamedTuple
import numpy as np
from BanditLearner import BanditLearner, BanditConfiguration
from parameters import products, environment


class EstimationParameter(NamedTuple):
    t_0: float = 0.0001  # Precision
    mu_0: float = 1  # Mean


class GaussianTSLearner(BanditLearner):
    def __init__(self, config: BanditConfiguration):
        super().__init__(config)
        self.Q = [0 for _ in products]  # estimated reward
        self.parameters = [[EstimationParameter() for _ in product.candidate_prices] for product in
                           products]  # precision

    def _select_price_indexes(self) -> List[int]:
        result = []
        for product in products:
            result.append(int(np.argmax([(np.random.randn() / (np.sqrt(self.parameters[product.id][i].t_0)) +
                                         self.parameters[product.id][i].mu_0) for i in
                                        range(len(product.candidate_prices))])))
        return result

    def _update(self):
        rewards = self.get_product_rewards()
        selected_price_indexes = self._history[-1][0]
        ### TODO: Why aren't we using customer count here?
        for p_i, reward in enumerate(rewards):
            self.Q[p_i] = (1 - 1.0 / environment.day) * self.Q[p_i] + (1.0 / environment.day) * reward
            i = selected_price_indexes[p_i]
            old_t, old_mu = self.parameters[p_i][i]
            new_mu = ((old_t * old_mu) + (environment.day * self.Q[p_i])) / (
                        old_t + environment.day)
            new_t = old_t + 1
            self.parameters[p_i][i] = EstimationParameter(t_0=new_t, mu_0=new_mu)
