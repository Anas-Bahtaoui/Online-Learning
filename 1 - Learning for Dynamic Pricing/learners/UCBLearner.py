from typing import List

import numpy as np

from BanditLearner import BanditLearner, BanditConfiguration
from entities import Product


class UCBLearner(BanditLearner):
    def _select_price_criteria(self, product: Product) -> List[float]:
        return np.array(self.means[product.id]) + np.array(self.widths[product.id])

    def reset(self):
        self.means = [[0 for _ in product.candidate_prices] for product in self._products]
        self.widths = [[np.inf for _ in product.candidate_prices] for product in self._products]
        self.rewards_per_arm_per_product = [[[] for _ in product.candidate_prices] for product in self._products]
        self.__init__(self.config)

    def __init__(self, config: BanditConfiguration):
        super().__init__(config)

    def _update(self):
        t = len(self._history)
        selected_price_indexes, last_customers = self._history[-1]
        for product_id, selected_price_index in enumerate(selected_price_indexes):
            for customer in last_customers:
                reward = self._get_reward_coef(customer, product_id) * self._products[product_id].candidate_prices[
                    selected_price_index]
                self.rewards_per_arm_per_product[product_id][selected_price_index].append(reward)
            self.means[product_id][selected_price_index] = np.mean(
                self.rewards_per_arm_per_product[product_id][selected_price_index])
            n = len(self.rewards_per_arm_per_product[product_id][selected_price_index])
            if n > 0:
                self.widths[product_id][selected_price_index] = np.sqrt(2 * np.log(t + 1) / n)
            else:
                self.widths[product_id][selected_price_index] = np.inf
