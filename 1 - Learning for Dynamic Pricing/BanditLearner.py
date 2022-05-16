from typing import List, Tuple, Optional, NamedTuple, Union

import numpy as np

from Customer import Customer, CustomerClass
from Learner import Learner, ShallContinue, Reward, PriceIndexes
from Product import Product, ObservationProbability
from parameters import environment, products, LAMBDA_, customer_counts, purchase_amounts


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


test_step = BanditConfiguration("Test Step", False, False, False)
step3 = BanditConfiguration("Step 3", True, True, True)
step4 = BanditConfiguration("Step 4", False, False, True)
step5 = BanditConfiguration("Step 5", True, True, False)
# TODO: Step 6 is unclear, what will the first 3 parameters be
# TODO: Ask to the professor
step6_sliding_window = BanditConfiguration("Step 6 with Sliding Window", False, False, False, 10)
step6_change_detection = BanditConfiguration("Step 6 with Custom Algorithm", False, False, False,
                                             ChangeDetectionAlgorithm())
step7 = BanditConfiguration("Step 7", False, False, True, None, (14, ContextGenerationAlgorithm()))


class BanditLearner(Learner):
    def iterate_once(self) -> Tuple[ShallContinue, Reward, PriceIndexes]:
        selected_price_indexes = self._run_one_day()
        reward = 0
        for customer in self._history[-1][1]:
            for productId in range(5):
                reward += self._get_reward_coef(customer, productId) * selected_price_indexes[productId]
        return True, reward, selected_price_indexes

    def get_product_rewards(self) -> List[float]:
        rewards = [0 for _ in products]
        for customer in self._history[-1][1]:
            for productId in range(5):
                rewards[productId] += self._get_reward_coef(customer, productId) * self._history[-1][0][productId]
        return rewards

    def _get_reward_coef(self, customer: Customer, product_id: int):
        return 1 if customer.products_bought[product_id] > 0 else 0
        #  return customer.products_bought[product_id]
        # TODO: Also here, why no product count?

    def calculate_reward(self, history_item: Tuple[List[int], List[Customer]]) -> List[float]:
        rewards = [0 for _ in products]
        selected_price_indexes, customers = history_item
        for customer in customers:
            for productId in range(5):
                rewards[productId] += self._get_reward_coef(customer, productId) * selected_price_indexes[productId]
        return rewards

    def _normalize_rewards(self, rewards: List[float]) -> List[float]:
        product_probabilities = [1 for _ in products]
        if self.config.a_ratios_known:
            # Eliminate alpha ratios, this is before graph weights
            # because alphas effect the graph weights.
            alpha_expectations = environment.alpha_distribution.get_expectation()
            total_alpha = sum(alpha_expectations)
            for i in range(len(product_probabilities)):
                product_probabilities[i] *= alpha_expectations[i] / total_alpha

        if self.config.graph_weights_known:
            # Eliminate graph weight effect
            accumulated_weights = [0 for _ in products]

            def run_on_product(product: Product, weight: float):
                if visited[product.id]:
                    return
                visited[product.id] = True
                accumulated_weights[product.id] += weight
                sp1, sp2 = product.secondary_products
                if sp1:
                    run_on_product(sp1[0], sp1[1] * weight)
                if sp2:
                    run_on_product(sp2[0], sp2[1] * weight * LAMBDA_)

            for product in products:
                visited = [False for _ in products]
                run_on_product(product, 1)
            total_weights = sum(accumulated_weights)
            for i in range(len(product_probabilities)):
                product_probabilities[i] *= accumulated_weights[i] / total_weights

        new_rewards = [reward * product_probability for reward, product_probability in
                       zip(rewards, product_probabilities)]
        if self.config.n_items_sold_known:
            # We normalize also by based on the amount of people
            for product in products:
                expected_purchase = 0
                for class_ in CustomerClass:
                    expected_purchase += purchase_amounts[class_][product.id].get_expectation() * customer_counts[
                        class_].get_expectation()
                new_rewards[product.id] /= expected_purchase
        return new_rewards

    def __init__(self, config: BanditConfiguration):
        self.name = f"{type(self).__name__} for {config.name}"
        self.config = config
        self.means = [[0 for _ in product.candidate_prices] for product in products]
        self.widths = [[np.inf for _ in product.candidate_prices] for product in products]
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
                reservation_price = round(
                    customer.get_reservation_price_of(product.id, product_price).get_sample_value(), 2)
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
                    customer_views_second_product = bool(np.random.binomial(1, second_p[1] * LAMBDA_))
                    if customer_views_second_product:
                        run_on_product(second_p[0])

            first_product = np.random.choice([None, *products], p=environment.get_current_alpha())
            if first_product is not None:
                run_on_product(first_product)
        self._history.append((selected_price_indexes, customers))
        ### TODO: Discard history based on non_stationary parameter

    def _update_from_history_item(self, len_history: int, history_item: Tuple[List[int], List[Customer]]):
        raise NotImplementedError()

    def _update(self):
        self._update_from_history_item(len(self._history), self._history[-1])

    def _run_one_day(self):
        environment.new_day()
        selected_price_indexes = self._select_price_indexes()
        self._new_day(selected_price_indexes)
        self._update()
        return selected_price_indexes

    def reset(self):
        self.__init__(self.config)

    def clairvoyant_reward(self):
        selected_price_indexes = self._select_price_indexes()
        expected_total_reward = 0
        for customer_class in list(CustomerClass):
            expected_customer = Customer(customer_class)
            expected_customer_count = customer_counts[customer_class].get_expectation()
            for product in products:
                expected_product_price = product.candidate_prices[selected_price_indexes[product.id]]
                expected_customer_reservation_price = expected_customer.get_reservation_price_of(product.id,
                                                                                                 expected_product_price).get_sample_value()
                if expected_product_price > expected_customer_reservation_price:
                    reward = expected_product_price * expected_customer_count
                    if self.are_counts_certain:
                        reward *= purchase_amounts[customer_class][product.id].get_expectation()
                    expected_total_reward += reward
        return expected_total_reward
