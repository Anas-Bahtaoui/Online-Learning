from typing import List, Tuple, Optional, NamedTuple, Union

import numpy as np

from Customer import Customer, purchase_amounts, CustomerClass, customer_counts
from Environment import Environment
from Learner import Learner, ShallContinue, Reward, PriceIndexes
from Product import Product, ObservationProbability


class ChangeDetectionAlgorithm:
    pass


class ContextGenerationAlgorithm:
    pass


class BanditConfiguration(NamedTuple):
    name: str
    a_ratios_known: bool
    n_items_sold_known: bool
    graph_weights_known: bool
    # Stationary, sliding window size, or change detection algorithm
    non_stationary: Union[None, int, ChangeDetectionAlgorithm] = None
    with_context: Optional[Tuple[int, ContextGenerationAlgorithm]] = None  # First is the amount of days


step3 = BanditConfiguration("Step 3", True, True, True)
step4 = BanditConfiguration("Step 4", False, False, True)
step5 = BanditConfiguration("Step 5", True, True, False)
# TODO: Step 6 is unclear, what will the first 3 parameters be?
step6_sliding_window = BanditConfiguration("Step 6 with Sliding Window", False, False, False, 10)
step6_change_detection = BanditConfiguration("Step 6 with Custom Algorithm", False, False, False,
                                             ChangeDetectionAlgorithm())
step7 = BanditConfiguration("Step 7", False, False, True, None, (14, ContextGenerationAlgorithm()))


def reward_for_certain_count(customer: Customer, product_id: int) -> float:
    ### For step 3, 5, 6
    return customer.products_bought[product_id]


def reward_for_uncertain_count(customer: Customer, product_id: int) -> float:
    ### For step 4 and step 7 we don't know the count of products bought
    return 1 if customer.products_bought[product_id] > 0 else 0


class BanditLearner(Learner):
    def iterate_once(self) -> Tuple[ShallContinue, Reward, PriceIndexes]:
        selected_price_indexes = self._run_one_day()
        reward = 0
        for customer in self._history[-1][1]:
            for productId in range(5):
                reward += self._get_reward(customer, productId)
        return True, reward, selected_price_indexes

    def __init__(self, env: Environment, config: BanditConfiguration):
        self.products = env.products
        self.env = env
        self.name = f"{type(self).__name__} for {config.name}"
        self.are_counts_certain = config.n_items_sold_known
        self.config = config
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
        return selected_price_indexes

    def run_for(self, days: int):
        for _ in range(days):
            self._run_one_day()

    def reset(self):
        self.__init__(self.env, self.config)

    def _get_reward(self, customer: Customer, product_id: int) -> float:
        if self.are_counts_certain:
            return reward_for_certain_count(customer, product_id)
        else:
            return reward_for_uncertain_count(customer, product_id)

    def clairvoyant_reward(self):
        selected_price_indexes = self._select_price_indexes()
        expected_total_reward = 0
        for customer_class in list(CustomerClass):
            expected_customer = Customer(customer_class)
            expected_customer_count = customer_counts[customer_class].get_expectation()
            for product in self.products:
                expected_product_price = product.candidate_prices[selected_price_indexes[product.id]]
                expected_customer_reservation_price = expected_customer.get_reservation_price_of(product.id,
                                                                                                 expected_product_price).get_sample_value()
                if expected_product_price > expected_customer_reservation_price:
                    reward = (expected_product_price - product.production_cost) * expected_customer_count
                    if self.are_counts_certain:
                        reward *= purchase_amounts[customer_class][product.id].get_expectation()
                    expected_total_reward += reward
        return expected_total_reward
