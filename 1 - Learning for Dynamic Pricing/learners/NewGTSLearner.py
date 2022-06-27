from typing import List

import numpy as np

from BanditLearner import BanditLearner, BanditConfiguration
from entities import Product, np_random


class NewGTSLearner(BanditLearner):
    def __init__(self, config: BanditConfiguration):
        super().__init__(config)

    def _select_price_criteria(self, product: Product) -> List[float]:
        return np_random.normal(self.means[product.id], self.sigmas[product.id])

    def _update_learner_state(self, selected_price_indexes, product_rewards, t):
        for product_id, product_reward in enumerate(product_rewards):
            pulled_arm = selected_price_indexes[product_id]
            self._rewards_per_arm[product_id][pulled_arm].append(product_reward)
            self._collected_rewards[product_id].append(product_reward)
            self.means[product_id] = np.mean(self._rewards_per_arm[product_id][pulled_arm])
            n_samples = len(self._rewards_per_arm[pulled_arm])
            if n_samples > 1:
                self.sigmas[product_id][pulled_arm] = np.std(self._rewards_per_arm[product_id][pulled_arm]) / n_samples

    def reset(self):
        n_products = len(self._config.product_configs)
        n_arms = len(self._config.product_configs[0].prices)
        self.means = [np.zeros(n_arms) for _ in range(n_products)]
        self.sigmas = [np.full(n_arms, 1e3) for _ in range(n_products)]
        self._rewards_per_arm = [[[] for _ in range(n_arms)] for _ in range(n_products)]
        self._collected_rewards = [[] for _ in range(n_products)]
        self.__init__(self.config)
