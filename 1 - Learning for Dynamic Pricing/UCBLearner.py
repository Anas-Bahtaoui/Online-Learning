import datetime
from typing import List, Tuple, Optional

import numpy as np

import sample
from Customer import Customer, purchase_amounts, CustomerClass, customer_counts
from Environment import Environment
from Product import Product, ObservationProbability


class Learner:
    def __init__(self, env: Environment):
        self.products = env.products
        self.env = env
        self.means = [[0 for _ in product.candidate_prices] for product in self.products]
        self.widths = [[np.inf for _ in product.candidate_prices] for product in self.products]
        self._history: List[Tuple[List[int], List[Customer]]] = []

    def _select_price_indexes(self) -> List[int]:
        raise NotImplementedError()

    def _new_day(self, selected_price_indexes: List[int]):
        """
        :param selected_price_indexes: List of price indexes that we select for this run, for each product
        """
        customers = []
        for customer_class in list(CustomerClass):
            customers.extend(
                [Customer(customer_class) for _ in range(customer_counts[customer_class].get_sample_value())])

        for customer in customers:
            def run_on_product(product: Product):
                if customer.is_product_clicked(product.id):
                    return
                customer.click_product(product.id)
                product_price = product.candidate_prices[selected_price_indexes[product.id]]
                reservation_price = customer.get_reservation_price_of(product.id, product_price).get_sample_value()
                if reservation_price < product_price:
                    return
                buy_count = purchase_amounts[customer.class_][product.id].get_sample_value()
                customer.buy_product(product.id, buy_count)
                first_p: Optional[ObservationProbability]
                second_p: Optional[ObservationProbability]
                first_p, second_p = product.secondary_products
                if first_p is not None:
                    customer_views_first_product = bool(np.random.binomial(1, first_p[1]))
                    if customer_views_first_product:
                        run_on_product(first_p[0])
                if second_p is not None:
                    customer_views_second_product = bool(np.random.binomial(1, second_p[1] * self.env.lambda_))
                    if customer_views_second_product:
                        run_on_product(second_p[0])

            first_product = np.random.choice([None, *self.products], p=self.env.get_current_alpha())
            if first_product is not None:
                run_on_product(first_product)
        self._history.append((selected_price_indexes, customers))

    def _update(self):
        raise NotImplementedError()

    def _run_one_day(self):
        self.env.new_day()
        selected_price_indexes = self._select_price_indexes()
        self._new_day(selected_price_indexes)
        self._update()

    def run_for(self, days: int):
        for _ in range(days):
            self._run_one_day()

    def reset(self):
        self.__init__(self.env)


class UCBLearner(Learner):
    def _select_price_indexes(self) -> List[int]:
        return [np.argmax(np.array(self.means[product.id]) + np.array(self.widths[product.id])) for product in
                self.products]

    def __init__(self, env: Environment):
        super().__init__(env)
        self.means = [[0 for _ in product.candidate_prices] for product in self.products]
        self.widths = [[np.inf for _ in product.candidate_prices] for product in self.products]
        self.rewards_per_arm_per_product = [[[] for _ in product.candidate_prices] for product in self.products]

    def _update(self):
        t = len(self._history)
        last_selection, last_customers = self._history[-1]
        for product_id, selected_price_index in enumerate(last_selection):
            for customer in last_customers:
                reward = 1 if customer.products_bought[product_id] > 0 else 0
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
    learner = UCBLearner(env)
    n_exp = 100
    for _ in range(n_exp):
        learner.run_for(100)
        learner.reset()
