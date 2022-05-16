from typing import List, Tuple

import numpy as np

from BanditLearner import BanditLearner, BanditConfiguration
from Customer import Customer
from parameters import products


class UCBLearner(BanditLearner):
    def _select_price_indexes(self) -> List[int]:
        ### TODO: We need to make a prediction based on alpha values expectancy vs unused alpha
        return [np.argmax(np.array(self.means[product.id]) + np.array(self.widths[product.id])) for product in
                products]

    def __init__(self, config: BanditConfiguration):
        super().__init__(config)
        self.means = [[0 for _ in product.candidate_prices] for product in products]
        self.widths = [[np.inf for _ in product.candidate_prices] for product in products]
        self.rewards_per_arm_per_product = [[[] for _ in product.candidate_prices] for product in products]

    def _update_from_history_item(self, len_history: int, history_item: Tuple[List[int], List[Customer]]):
        t = len_history
        last_selection, last_customers = history_item
        product_mean_rewards = [0 for _ in products]
        for product_id, selected_price_index in enumerate(last_selection):
            for customer in last_customers:
                reward = self._get_reward_coef(customer, product_id) * products[product_id].candidate_prices[selected_price_index]
                self.rewards_per_arm_per_product[product_id][selected_price_index].append(reward)
            product_mean_rewards[product_id] = np.mean(self.rewards_per_arm_per_product[product_id][selected_price_index])
        product_mean_rewards = self._normalize_rewards(product_mean_rewards)
        for product_id, selected_price_index in enumerate(last_selection):
            self.means[product_id][selected_price_index] = product_mean_rewards[selected_price_index]
            n = len(self.rewards_per_arm_per_product[product_id][selected_price_index])
            if n > 0:
                self.widths[product_id][selected_price_index] = np.sqrt(2 * np.log(t + 1) / n)
            else:
                self.widths[product_id][selected_price_index] = np.inf