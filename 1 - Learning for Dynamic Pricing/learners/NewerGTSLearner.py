from typing import List

import numpy as np

from BanditLearner import BanditLearner, BanditConfiguration
from entities import Product, np_random


class NewerGTSLearner(BanditLearner):
    def _upper_bound(self):
        n_arms = len(self._rewards_per_arm[0])
        return self._C * np.sqrt(n_arms * self.total_days * np.log(n_arms))

    def update_experiment_days(self, days: int):
        self.total_days = days

    def _reset_parameters(self):
        n_products = len(self._config.product_configs)
        n_arms = len(self._config.product_configs[0].prices)
        self._precision = 1
        self.mu_0s = [np.full(n_arms, 1) for _ in range(n_products)]
        self.tau_0s = [np.full(n_arms, 1e-4) for _ in range(n_products)]
        self._rewards_per_arm = [[[] for _ in range(n_arms)] for _ in range(n_products)]
        self._C = 100

    def __init__(self, config: BanditConfiguration):
        super().__init__(config)

    def _select_price_criteria(self, product: Product) -> List[float]:
        return np_random.standard_normal() / np.sqrt(self.tau_0s[product.id]) + self.mu_0s[product.id]

    def _update_learner_state(self, selected_price_indexes, product_rewards, t):
        for product_id, product_reward in enumerate(product_rewards):
            pulled_arm = selected_price_indexes[product_id]
            self._rewards_per_arm[product_id][pulled_arm].append(product_reward)
            n_samples = len(self._rewards_per_arm[pulled_arm])
            n_tested = len(self._rewards_per_arm[product_id][pulled_arm])
            tau_0 = self.tau_0s[product_id][pulled_arm]
            mu_0 = self.mu_0s[product_id][pulled_arm]
            if n_samples > 1:
                self.tau_0s[product_id][pulled_arm] = tau_0 + n_tested * self._precision

            self.mu_0s[product_id][pulled_arm] = (tau_0 * mu_0 + np.sum(
                self._rewards_per_arm[product_id][pulled_arm]) * self._precision) / (tau_0 + n_tested * self._precision)
