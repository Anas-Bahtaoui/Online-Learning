from typing import List

import numpy as np

import sample
from BanditLearner import BanditLearner, BanditConfiguration, step3
from Environment import Environment


class UCBLearner(BanditLearner):
    def _select_price_indexes(self) -> List[int]:
        return [np.argmax(np.array(self.means[product.id]) + np.array(self.widths[product.id])) for product in
                self.products]

    def __init__(self, env: Environment, config: BanditConfiguration):
        super().__init__(env, config)
        self.means = [[0 for _ in product.candidate_prices] for product in self.products]
        self.widths = [[np.inf for _ in product.candidate_prices] for product in self.products]
        self.rewards_per_arm_per_product = [[[] for _ in product.candidate_prices] for product in self.products]

    def _update(self):
        t = len(self._history)
        last_selection, last_customers = self._history[-1]
        for product_id, selected_price_index in enumerate(last_selection):
            for customer in last_customers:
                reward = self._get_reward(customer, product_id)
                self.rewards_per_arm_per_product[product_id][selected_price_index].append(reward)
            self.means[product_id][selected_price_index] = int(
                np.mean(self.rewards_per_arm_per_product[product_id][selected_price_index]))
            n = len(self.rewards_per_arm_per_product[product_id][selected_price_index])
            if n > 0:
                self.widths[product_id][selected_price_index] = np.sqrt(2 * np.log(t + 1) / n)
            else:
                self.widths[product_id][selected_price_index] = np.inf


if __name__ == '__main__':
    env = sample.generate_sample_greedy_environment()
    learner = UCBLearner(env, step3)
    n_exp = 4
    prices_selected = []
    last_prices_selected = None
    n = 0
    for _ in range(n_exp):
        n += 1
        learner.run_for(400)
        prices_selected.append(learner._history[-1][0])
        if last_prices_selected == prices_selected[-1]:
            break
        last_prices_selected = prices_selected[-1]
        learner.reset()
    print(prices_selected)
    print(n)
