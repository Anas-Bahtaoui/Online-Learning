from collections import defaultdict
from typing import List, Tuple, Optional, NamedTuple, Union
import numpy as np

from Learner import Learner, ShallContinue, Reward, PriceIndexes
from entities import CustomerClass, Product, ObservationProbability, Customer


class ChangeDetectionAlgorithm:
    pass


class ContextGenerationAlgorithm:
    pass


# TODO: Things we don't know, use the empirical mean
# TODO: Things we know use the expectation of distribution.

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
# TODO: Step 6 is unclear, what will the first 3 parameters be
# TODO: Ask to the professor
step6_sliding_window = BanditConfiguration("Step 6 with Sliding Window", False, False, False, 10)
step6_change_detection = BanditConfiguration("Step 6 with Custom Algorithm", False, False, False,
                                             ChangeDetectionAlgorithm())
step7 = BanditConfiguration("Step 7", False, False, True, None, (14, ContextGenerationAlgorithm()))


def reward_multiplier_for_certain_count(customer: Customer, product_id: int) -> float:
    ### For step 3, 5, 6
    return customer.products_bought[product_id]


def reward_multiplier_for_uncertain_count(customer: Customer, product_id: int) -> float:
    ### For step 4 and step 7 we don't know the count of products bought
    return 1 if customer.products_bought[product_id] > 0 else 0


class BanditLearner(Learner):
    def iterate_once(self) -> Tuple[ShallContinue, Reward, PriceIndexes]:
        self._run_one_day()
        selected_price_indexes, customers = self._history[-1]
        reward = 0
        for customer in customers:
            for product in self._products:
                reward += self._get_reward_coef(customer, product.id) * product.candidate_prices[
                    selected_price_indexes[product.id]]
        return True, reward, selected_price_indexes

    def get_product_rewards(self) -> List[float]:
        rewards = [0 for _ in self._products]
        selected_price_indexes, customers = self._history[-1]
        for customer in customers:
            for product in self._products:
                rewards[product.id] += self._get_reward_coef(customer, product.id) * product.candidate_prices[
                    selected_price_indexes[product.id]]
        for product in self._products:
            for class_ in CustomerClass:
                inter = 0
                custs_ = [customer for customer in customers if customer.class_ == class_]
                cnt = 0
                clicked = 0
                total_reservation_price = 0
                for customer in custs_:
                    inter += self._get_reward_coef(customer, product.id) * product.candidate_prices[
                        selected_price_indexes[product.id]]
                    cnt += self._get_reward_coef(customer, product.id)
                    # total_reservation_price += customer.get_reservation_price_of(product.id,
                    #                                                          selected_price_indexes[product.id])
                    clicked += 1 if customer.is_product_clicked(product.id) else 0
                if self._verbose:
                    print("For product", product.name, "user class", class_, "selected index",
                          selected_price_indexes[product.id],
                          "Actual reward:", inter, "Actual amount of customers", len(custs_),
                          "Actual clicked #customers", clicked, "Actual bought #customers", cnt)
        return rewards

    def __init__(self, config: BanditConfiguration):
        super().__init__()
        self.name = f"{type(self).__name__} for {config.name}"
        self.are_counts_certain = False  # config.n_items_sold_known
        self.config = config
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
                [Customer(customer_class) for _ in
                 range(self._config.customer_counts[customer_class].get_sample_value())])
        total_reservation_prices = defaultdict(list)
        for customer in customers:
            def run_on_product(product: Product):
                if customer.is_product_clicked(product.id):
                    return
                customer.click_product(product.id)
                product_price = product.candidate_prices[selected_price_indexes[product.id]]
                reservation_price = round(
                    customer.get_reservation_price_of(product.id, product_price).get_sample_value(), 2)
                if reservation_price < product_price:
                    return reservation_price
                buy_count = self._config.purchase_amounts[customer.class_][product.id].get_sample_value()
                customer.buy_product(product.id, buy_count)
                first_p: Optional[ObservationProbability]
                second_p: Optional[ObservationProbability]
                first_p, second_p = product.secondary_products[customer.class_]
                if first_p is not None:
                    customer_views_first_product = bool(np.random.binomial(1, first_p[1]))
                    if customer_views_first_product:
                        run_on_product(first_p[0])
                if second_p is not None:
                    customer_views_second_product = bool(np.random.binomial(1, second_p[1] * self._config.lambda_))
                    if customer_views_second_product:
                        run_on_product(second_p[0])
                return reservation_price

            first_product = np.random.choice([None, *self._products],
                                             p=self._environment.get_current_alpha(customer.class_))
            if first_product is not None:
                total_reservation_prices[(customer.class_, first_product.id)].append(run_on_product(first_product))
        for product in self._products:
            for class_ in CustomerClass:
                prices = total_reservation_prices[(class_, product.id)]
                if len(prices) == 0:
                    continue
                if self._verbose:
                    print("Average reservation price for product", product.name, "for class", class_,
                          sum(prices) / len(prices))
        self._history.append((selected_price_indexes, customers))

    def _update(self):
        raise NotImplementedError()

    def _run_one_day(self):
        self._environment.new_day()
        selected_price_indexes = self._select_price_indexes()
        self._new_day(selected_price_indexes)
        self._update()

    def reset(self):
        self.means = [[0 for _ in product.candidate_prices] for product in self._products]
        self.widths = [[np.inf for _ in product.candidate_prices] for product in self._products]
        self.__init__(self.config)

    def _get_reward_coef(self, customer: Customer, product_id: int) -> float:
        # if self.are_counts_certain:
        #     return reward_multiplier_for_certain_count(customer, product_id)
        # else:
        return reward_multiplier_for_uncertain_count(customer, product_id)

    def clairvoyant_reward(self):
        selected_price_indexes = self._select_price_indexes()
        expected_total_reward = 0
        for customer_class in list(CustomerClass):
            expected_customer = Customer(customer_class)
            expected_customer_count = self._config.customer_counts[customer_class].get_expectation()
            for product in self._products:
                expected_product_price = product.candidate_prices[selected_price_indexes[product.id]]
                expected_customer_reservation_price = expected_customer.get_reservation_price_of(product.id,
                                                                                                 expected_product_price).get_sample_value()
                if expected_product_price > expected_customer_reservation_price:
                    reward = expected_product_price * expected_customer_count
                    if self.are_counts_certain:
                        reward *= self._config.purchase_amounts[customer_class][product.id].get_expectation()
                    expected_total_reward += reward
        return expected_total_reward
