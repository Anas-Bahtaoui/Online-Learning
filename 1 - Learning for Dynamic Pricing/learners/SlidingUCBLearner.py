'''
- Add the sliding window to the init
- Add the sliding window to the reset
- Add a new funtion that if the age of the arm is greater than the max number of days we want to keep, we delete the oldest data of the arm -> Is this already implemented by Ozan?
- Add the sliding window to the update
'''
import math
from typing import List

import numpy as np

from BanditLearner import BanditLearner, BanditConfiguration
from entities import Product


class SlidingUCBLearner(BanditLearner):
    def _reset_parameters(self):
        self._cache = []
        self._reset_ucb_params()

    def _select_price_criteria(self, product: Product) -> List[float]:
        return np.array(self.means[product.id]) + np.array(self.widths[product.id])

    def __init__(self, config: BanditConfiguration):
        super().__init__(config)
        # TODO Add the sliding window to the init, but how?
    def _reset_ucb_params(self):
        self.means = [[0 for _ in product.candidate_prices] for product in self._products]
        self.widths = [[np.inf for _ in product.candidate_prices] for product in self._products]
        self.rewards_per_arm_per_product = [[[] for _ in product.candidate_prices] for product in self._products]
    def update_experiment_days(self, days: int):
        super().update_experiment_days(days)
        self._window_size = int(days ** 0.5)

    def _update_learner_state(self, selected_price_indexes_, product_rewards_, t_):
        self._cache.append((selected_price_indexes_, product_rewards_))
        self._reset_ucb_params()
        for ind, (selected_price_indexes, product_rewards) in enumerate(self._cache[-self._window_size:]):
            t = ind + 1 # T starts from 1
            for product_id, product_reward in enumerate(product_rewards):
                pulled_arm = selected_price_indexes[product_id]

                self.rewards_per_arm_per_product[product_id][pulled_arm].append(product_reward)

                n = len(self.rewards_per_arm_per_product[product_id][pulled_arm])
                if n > 0:
                    self.means[product_id][pulled_arm] = np.mean(self.rewards_per_arm_per_product[product_id][pulled_arm])
                    self.widths[product_id][pulled_arm] = np.sqrt(2 * np.log(t + 1) / n)
                else:
                    self.means[product_id][pulled_arm] = 0
                    self.widths[product_id][pulled_arm] = np.inf
