from typing import List, NamedTuple

from BanditLearner import BanditLearner, BanditConfiguration
import numpy as np

from parameters import products


class BetaParameters(NamedTuple):
    alpha: float
    beta: float


class TSLearner(BanditLearner):
    def __init__(self, config: BanditConfiguration):
        super().__init__(config)
        self.beta_params: List[List[BetaParameters]] = [
            [BetaParameters(alpha=1, beta=1) for _ in product.candidate_prices] for product in
            products]

    def _select_price_indexes(self) -> List[int]:
        result = []
        for product in products:
            result.append(int(np.argmax(np.random.beta(
                [param.alpha for param in self.beta_params[product.id]],
                [param.beta for param in self.beta_params[product.id]]
            ))))
        return result

    def _update(self):
        last_selection, last_customers = self._history[-1]
        for product_id, selected_price_index in enumerate(last_selection):
            ### TODO: This doesn't vary by how many products they bought
            # Either find a way to represent this in an alpha beta distribution (how to know how many products they didn't buy?)
            # Or just trash the whole number of product idea and reinterpret wth it says in step 4 about product counts.

            bought, not_bought = 0, 0
            for customer in last_customers:
                if customer.products_bought[product_id] > 0:
                    # Shall alpha and beta update based on reward or just binary variable or product count?
                    bought += 1 # Have the product counts here?
                else:
                    not_bought += 1

            old_alpha, old_beta = self.beta_params[product_id][selected_price_index]

            new_alpha = old_alpha + bought
            new_beta = old_beta + not_bought

            self.beta_params[product_id][selected_price_index] = BetaParameters(new_alpha, new_beta)

