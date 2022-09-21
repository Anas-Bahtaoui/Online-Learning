from collections import defaultdict
from typing import Optional
import numpy as np

from Learner import Learner, ShallContinue, ExperimentHistoryItem, Reward, CR
from change_detectors import ChangeDetectionAlgorithm, CumSum
from entities import Environment, SimulationConfig, np_random
from parameter_estimators import *


# TODO: Things we don't know, use the empirical mean
# TODO: Things we know use the expectation of distribution.

class NonStationaryConfig(NamedTuple):
    normal_days: int
    abrupt_days: int
    change_detection_algorithm: Optional[ChangeDetectionAlgorithm]


class BanditConfiguration(NamedTuple):
    name: str
    a_ratios_known: bool
    n_items_sold_known: bool
    graph_weights_known: bool
    # Stationary, sliding window size, or change detection algorithm
    non_stationary: Optional[NonStationaryConfig] = None
    context_generation: bool = False  # First is the amount of days


step3 = BanditConfiguration("Step 3", True, True, True)
step4 = BanditConfiguration("Step 4", False, False, True)
step5 = BanditConfiguration("Step 5", True, True, False)
# Step 6 is there for comparing two algorithms, it is better to just say all is unknown
step6_sliding_window = BanditConfiguration("Step 6 with Sliding Window", False, False, False,
                                           NonStationaryConfig(30, 30, None))
step6_change_detection = BanditConfiguration("Step 6 with Custom Algorithm", False, False, False,
                                             NonStationaryConfig(30, 30, CumSum(10, 0.1, 0.2)))
step7 = BanditConfiguration("Step 7", False, False, True, None, True)


class BanditLearner(Learner):
    def iterate_once(self) -> ShallContinue:
        self._run_one_day()
        return True  # Bandit learner always runs

    def _calculate_product_rewards(self, selected_price_indexes, customers: List["Customer"]) -> List[float]:
        rewards = [0 for _ in self._products]
        for customer in customers:
            for product in self._products:
                rewards[product.id] += product.candidate_prices[selected_price_indexes[product.id]] * \
                                       customer.products_bought[product.id][0]

        return rewards

    def _calculate_product_crs(self, customers: List["Customer"]) -> List[float]:
        clicks = [0 for _ in self._products]
        buys = [0 for _ in self._products]
        for customer in customers:
            for product in self._products:
                clicks[product.id] += int(product.id in customer.products_clicked)
                buys[product.id] += 1 if customer.products_bought[product.id][0] > 0 else 0

        return [clicks[i] / buys[i] if buys[i] > 0 else 0 for i in range(len(self._products))]

    def __init__(self, config: BanditConfiguration):
        super().__init__()
        self.name = f"{type(self).__name__} for {config.name}"
        self.config = config
        if not hasattr(self, "_estimators"):
            self._estimators: List[ParameterEstimator] = []
        self._t = 0
        self._sliding_window_slide_period: Optional[int] = None

    def _shall_abrupt_change(self) -> bool:
        if self.config.non_stationary is None:
            return False
        period = self.config.non_stationary.normal_days + self.config.non_stationary.abrupt_days
        return (len(self._experiment_history) % period) >= self.config.non_stationary.normal_days

    def refresh_vars(self, products: List[Product], environment: Environment, config: SimulationConfig):
        super().refresh_vars(products, environment, config)
        self._experiment_history: List[ExperimentHistoryItem] = []
        self._reset_parameters()
        if self.config.non_stationary is not None and self.config.non_stationary.change_detection_algorithm is not None:
            self.config.non_stationary.change_detection_algorithm.reset()
        self._t = 0

        self._estimators = []
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

    def update_experiment_days(self, days: int):
        if self.config.non_stationary is not None and self.config.non_stationary.change_detection_algorithm is None:
            self._sliding_window_slide_period = int(days ** 0.5)
        if self.config.non_stationary is not None and self.config.non_stationary.change_detection_algorithm is not None:
            self.config.non_stationary.change_detection_algorithm.update_experiment_days(days)

    def _select_price_indexes(self) -> List[int]:
        result = []
        for product in self._products:
            price_vals = self._select_price_criteria(product)
            selected_index = int(np.argmax(price_vals))
            result.append(selected_index)
        return result

    def _select_price_criteria(self, product: Product) -> List[float]:
        raise NotImplementedError()

    def _new_day(self, selected_price_indexes: List[int], persist=True):
        """
        :param selected_price_indexes: List of price indexes that we select for this run, for each product
        :param persist: If true, the results will be saved in the experiment history, otherwise product rewards and CRs will be returned
        This is needed for clairvoyant to work
        """
        customers = []
        abrupt_change = self._shall_abrupt_change()
        for customer_class in list(CustomerClass):
            customers.extend(
                [Customer(customer_class, is_abrupt=abrupt_change) for _ in
                 range(self._config.customer_counts[customer_class].get_sample_value())])
        for customer in customers:
            def run_on_product(product: Product):
                if customer.is_product_clicked(product.id):
                    return
                customer.click_product(product.id)
                product_price = product.candidate_prices[selected_price_indexes[product.id]]
                cr = customer.get_conversion_rate_for(product.id, product_price).get_sample_value()
                customer.see_product(product.id, cr)
                if cr < 0 or np.isnan(cr):
                    cr = 0
                if cr > 1:
                    cr = 1
                will_buy = bool(np_random.binomial(1, cr))
                if not will_buy:
                    return
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

            first_product = np_random.choice([None, *self._products],
                                             p=self._environment.get_current_alpha(customer.class_))
            if first_product is not None:
                run_on_product(first_product)
        product_rewards = self._calculate_product_rewards(selected_price_indexes, customers)
        product_crs = self._calculate_product_crs(customers)
        if persist:
            clairvoyant_rewards = self._clairvoyant_reward_calculate(self.clairvoyant_indexes)[0]
            self._experiment_history.append(
                ExperimentHistoryItem(sum(product_rewards), selected_price_indexes, product_rewards, product_crs, False,
                                      None,
                                      sum(clairvoyant_rewards), customers,
                                      {estimator.__class__.__name__: {} for estimator in self._estimators}, 0,
                                      self._shall_abrupt_change()))
            upper_bound = self._upper_bound()
            self._experiment_history[-1] = self._experiment_history[-1]._replace(upper_bound=upper_bound)
        else:
            return product_rewards, product_crs

    def _update_learner_state(self, selected_price_indexes, product_crs, t):
        raise NotImplementedError()

    def _reset_and_rerun_for_last_n(self, n: int):
        if n >= len(self._experiment_history):
            return
        self._t = 0
        self._reset_parameters()
        for estimator in self._estimators:
            estimator.reset()
        for i in range(-n, 0, 1):
            self._t += 1
            selected_price_indexes, product_crs = self._experiment_history[i].selected_price_indexes, \
                                                  self._experiment_history[i].product_crs
            if self._t > 1:
                last_customers = self._experiment_history[i - 1].customers
                for estimator in self._estimators:
                    for customer in last_customers:
                        estimator.update(customer)
                    product_crs = estimator.modify(product_crs, register_history=False)
            self._update_learner_state(selected_price_indexes, product_crs, self._t)

    def _update(self):
        self._t += 1
        selected_price_indexes, product_crs = self._experiment_history[-1].selected_price_indexes, \
                                              self._experiment_history[-1].product_crs
        if self._t > 1:
            for estimator in self._estimators:
                product_crs = estimator.modify(product_crs)

            self._experiment_history[-1] = self._experiment_history[-1]._replace(
                estimators={estimator.__class__.__name__: estimator._history[-1] for estimator in self._estimators})
        self._update_learner_state(selected_price_indexes, product_crs, self._t)

    def _update_parameter_estimators(self):
        customers = self._experiment_history[-1].customers
        for customer in customers:
            [estimator.update(customer) for estimator in self._estimators]

    def _run_one_day(self):
        self._environment.new_day()
        selected_price_indexes = self._select_price_indexes()
        self._new_day(selected_price_indexes)
        self._update_parameter_estimators()
        if self.config.non_stationary is not None:
            if self.config.non_stationary.change_detection_algorithm is None:
                self._reset_and_rerun_for_last_n(self._sliding_window_slide_period)
            else:
                algorithm = self.config.non_stationary.change_detection_algorithm
                non_stationary_result = algorithm.update(self._experiment_history[-1].customers)
                if algorithm.has_changed():
                    algorithm.reset()
                    self._reset_parameters()
                    self._reset_and_rerun_for_last_n(1)
                    self._experiment_history[-1] = self._experiment_history[-1][:3] + (True,) + \
                                                   self._experiment_history[
                                                       -1][4:]
                else:
                    self._update()
                self._experiment_history[-1] = self._experiment_history[-1][:4] + (non_stationary_result,) + \
                                               self._experiment_history[-1][5:]

        else:
            self._update()

    def reset(self):
        self.__init__(self.config)

    def _reset_parameters(self):
        raise NotImplementedError()

    def _clairvoyant_reward_calculate(self, price_indexes) -> Tuple[List[Reward], List[CR]]:
        return self._new_day(price_indexes, persist=False)

    """
    Might not work ...
    
    TODO We will write a new one similiar to the GREEDY but w/o the constraints.
    For each customer class calculate it and sum it up. Then get the maximum reward of all price indexes.
    """
