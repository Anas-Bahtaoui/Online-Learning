from typing import List

import numpy as np

from BanditLearner import BanditLearner, BanditConfiguration
from parameters import products


class UCBLearner(BanditLearner):
    def _select_price_indexes(self) -> List[int]:
        return [np.argmax(np.array(self.means[product.id]) + np.array(self.widths[product.id])) for product in
                products]

    def __init__(self, config: BanditConfiguration):
        super().__init__(config)
        self.means = [[0 for _ in product.candidate_prices] for product in products]
        self.widths = [[np.inf for _ in product.candidate_prices] for product in products]
        self.rewards_per_arm_per_product = [[[] for _ in product.candidate_prices] for product in products]

    def _update(self):
        t = len(self._history)
        last_selection, last_customers = self._history[-1]
        for product_id, selected_price_index in enumerate(last_selection):
            for customer in last_customers:
                reward = self._get_reward(customer, product_id)
                self.rewards_per_arm_per_product[product_id][selected_price_index].append(reward)
            self.means[product_id][selected_price_index] = int( # TODO: Why cast to int? There is something fishy or?
                np.mean(self.rewards_per_arm_per_product[product_id][selected_price_index]))
            n = len(self.rewards_per_arm_per_product[product_id][selected_price_index])
            if n > 0:
                self.widths[product_id][selected_price_index] = np.sqrt(2 * np.log(t + 1) / n)
            else:
                self.widths[product_id][selected_price_index] = np.inf


