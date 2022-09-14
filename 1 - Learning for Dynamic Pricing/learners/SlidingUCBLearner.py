'''
- Add the sliding window to the init
- Add the sliding window to the reset
- Add a new funtion that if the age of the arm is greater than the max number of days we want to keep, we delete the oldest data of the arm -> Is this already implemented by Ozan?
- Add the sliding window to the update
'''

from typing import List

import numpy as np

from BanditLearner import BanditLearner, BanditConfiguration
from entities import Product


class UCBLearner(BanditLearner):
    def _reset_parameters(self):
        self.means = [[0 for _ in product.candidate_prices] for product in self._products]
        self.widths = [[np.inf for _ in product.candidate_prices] for product in self._products]
        self.rewards_per_arm_per_product = [[[] for _ in product.candidate_prices] for product in self._products]
        # Add the sliding window to the reset
        self.sliding_window = [[0 for _ in product.candidate_prices] for product in self._products]

    def _select_price_criteria(self, product: Product) -> List[float]:
        return np.array(self.means[product.id]) + np.array(self.widths[product.id])

    def __init__(self, config: BanditConfiguration):
        super().__init__(config)
        # TODO Add the sliding window to the init, but how?


    def _update_learner_state(self, selected_price_indexes, product_rewards, t):
        for product_id, product_reward in enumerate(product_rewards):
            pulled_arm = selected_price_indexes[product_id]
            
            self.rewards_per_arm_per_product[product_id][pulled_arm].append(product_reward)
            # TODO Forget the rewards that are older than the sliding window
            
            n = len(self.rewards_per_arm_per_product[product_id][pulled_arm])
            if n > 0:
                self.means[product_id][pulled_arm] = np.mean(self.rewards_per_arm_per_product[product_id][pulled_arm])
                self.widths[product_id][pulled_arm] = np.sqrt(2 * np.log(t + 1) / n)
            else:
                self.means[product_id][pulled_arm] = 0
                self.widths[product_id][pulled_arm] = np.inf
