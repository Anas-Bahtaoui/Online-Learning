from collections import defaultdict
from typing import NamedTuple, Union, Optional
import numpy as np

from Learner import Learner, ShallContinue
from entities import Environment, SimulationConfig, np_random
from parameter_estimators import *


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


class BanditLearner(Learner):
    def iterate_once(self) -> ShallContinue:
        self._run_one_day()
        return True  # Bandit learner always runs

    def _calculate_product_rewards(self, selected_price_indexes) -> List[float]:
        rewards = [0 for _ in self._products]
        customers = self._customer_history[-1]
        for customer in customers:
            for product in self._products:
                rewards[product.id] += product.candidate_prices[selected_price_indexes[product.id]] * \
                                       customer.products_bought[product.id]

        return rewards

    def log_experiment(self):
        if self._verbose:
            _, selected_price_indexes, product_rewards = self._experiment_history[-1]
            customers = self._customer_history[-1]
            for product in self._products:
                purchase_count = 0
                product_click_count = 0
                for class_ in CustomerClass:
                    custs_ = [customer for customer in customers if customer.class_ == class_]
                    for customer in custs_:
                        purchase_count += customer.products_bought[product.id]
                        product_click_count += 1 if customer.is_product_clicked(product.id) else 0
                print("For product", product.name, "selected index",
                      selected_price_indexes[product.id],
                      "Actual reward:", product_rewards[product.id], "Actual amount of customers", len(customers),
                      "Actual clicked #customers", product_click_count, "Actual bought #customers", purchase_count)

    def __init__(self, config: BanditConfiguration):
        super().__init__()
        self.name = f"{type(self).__name__} for {config.name}"
        self.config = config
        self._customer_history: List[List[Customer]] = []
        if not hasattr(self, "_estimators"):
            self._estimators: List[ParameterEstimator] = []

    def set_vars(self, products: List[Product], environment: Environment, config: SimulationConfig):
        super().set_vars(products, environment, config)
        if self.config.a_ratios_known:
            # Calculate Customer class independent alphas

            alpha_prediction = self._environment.get_aggregate_alpha(self._config.customer_counts)
            self._estimators.append(KnownAlphaEstimator(alpha_prediction))
        else:
            self._estimators.append(AlphaEstimator())

        if self.config.n_items_sold_known:
            # Calculate customer class independent number of items sold
            self._estimators.append(
                KnownItemsSoldEstimator(self._config.customer_counts, self._config.purchase_amounts))
        else:
            self._estimators.append(NumberOfItemsSoldEstimator())

        if self.config.graph_weights_known:
            self._estimators.append(KnownGraphWeightsEstimator(self._config.secondaries, self._config.customer_counts,
                                                               self._config.lambda_))
        else:
            self._estimators.append(GraphWeightsEstimator())

    def _select_price_indexes(self) -> List[int]:
        result = []
        for product in self._products:
            price_vals = self._select_price_criteria(product)
            for estimator in self._estimators:
                assert all(value >= 0 for value in price_vals)
                price_vals = estimator.modify(price_vals)
                assert all(value >= 0 for value in price_vals)
            selected_index = int(np.argmax(price_vals))
            result.append(selected_index)
        return result

    def _select_price_criteria(self, product: Product) -> List[float]:
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
                    customer_views_first_product = bool(np_random.binomial(1, first_p[1]))
                    if customer_views_first_product:
                        run_on_product(first_p[0])
                if second_p is not None:
                    customer_views_second_product = bool(np_random.binomial(1, second_p[1] * self._config.lambda_))
                    if customer_views_second_product:
                        run_on_product(second_p[0])
                return reservation_price

            first_product = np_random.choice([None, *self._products],
                                             p=self._environment.get_current_alpha(customer.class_))
            if first_product is not None:
                total_reservation_prices[(customer.class_, first_product.id)].append(run_on_product(first_product))
        if self._verbose:
            for product in self._products:
                for class_ in CustomerClass:
                    prices = total_reservation_prices[(class_, product.id)]
                    if len(prices) == 0:
                        continue
                    print("Average reservation price for product", product.name, "for class", class_,
                          sum(prices) / len(prices))

        self._customer_history.append(customers)
        product_rewards = self._calculate_product_rewards(selected_price_indexes)
        self._experiment_history.append((sum(product_rewards), selected_price_indexes, product_rewards))

    def _update(self):
        raise NotImplementedError()

    def _update_parameter_estimators(self):
        customers = self._customer_history[-1]

        for customer in customers:
            _ = (estimator.update(customer) for estimator in self._estimators)

    def _run_one_day(self):
        self._environment.new_day()
        selected_price_indexes = self._select_price_indexes()
        self._new_day(selected_price_indexes)
        self._update()
        self._update_parameter_estimators()

    def reset(self):
        self.__init__(self.config)

    """
    Might not work ...
    
    TODO We will write a new one similiar to the GREEDY but w/o the constraints.
    For each customer class calculate it and sum it up. Then get the maximum reward of all price indexes.
    """

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
                    reward *= self._config.purchase_amounts[customer_class][product.id].get_expectation()
                    expected_total_reward += reward
        return expected_total_reward
