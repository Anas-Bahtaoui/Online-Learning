from operator import itemgetter
from typing import Tuple, Optional, List, Dict

import scipy.stats

from Learner import Learner, ShallContinue, Reward, PriceIndexes, ExperimentHistoryItem, CR
from entities import Product, ObservationProbability, CustomerClass, conversion_rate_distribution_from_curves


class GreedyLearner(Learner):
    def _upper_bound(self):
        return self.absolute_clairvoyant

    def update_experiment_days(self, days: int):
        pass

    def reset(self):
        self.__init__()

    def _get_rewards_of_current_run(self) -> List[float]:
        self._counts = {product.id: [0, 0] for product in self._products} # 0 is clicked, 1 is bought
        return [sum(
            self.calculate_reward_of_product(self.candidate_price_indexes[i], self._products[i], class_) for class_ in
            list(CustomerClass)) for i in
            range(len(self._products))]

    name = "Greedy Algorithm"

    def iterate_once(self) -> ShallContinue:
        shall_continue = self._iterate_once()
        product_rewards = self._get_rewards_of_current_run()
        product_crs = [self._counts[product.id][1] / self._counts[product.id][0] for product in self._products]
        clairvoyant = self._clairvoyant_reward_calculate(self.clairvoyant_indexes)[1]
        self._experiment_history.append(
            ExperimentHistoryItem(self.current_cr, list(self.candidate_price_indexes), product_rewards, product_crs, False, None,
                                  sum(clairvoyant), None, None, self._upper_bound(), False))
        return shall_continue

    def __init__(self):
        super(GreedyLearner, self).__init__()
        self.candidate_price_indexes = (0, 0, 0, 0, 0)
        self.current_cr = 0
        self._counts = {}

    # We need to pass the price indexes of all products
    def calculate_reward_of_product(self, price_index: int, product: Product, class_: CustomerClass) -> float:
        current_price_indexes = self.candidate_price_indexes[:product.id] + (
            price_index,) + self.candidate_price_indexes[product.id + 1:]
        n_users = self._config.customer_counts[class_].get_expectation()

        def emulate_path(clicked_primaries: Tuple[int, ...], viewing_probability: float, current: Product) -> Reward:
            """
            Side effect: fills in the clicks buys dictionary
            :param clicked_primaries:
            :param viewing_probability:
            :param current:
            :return:
            """
            # We already looked at this product, thus we can skip it
            if current.id in clicked_primaries:
                return 0

            product_price = current.candidate_prices[current_price_indexes[current.id]]
            # We don't have simulated users but use expected values directly
            # Greedy never uses abrupt change graphs
            reservation_price_distribution = conversion_rate_distribution_from_curves(class_, current.id,
                                                                                        product_price)
            purchase_ratio = reservation_price_distribution.get_expectation()
            self._counts[current.id][0] += viewing_probability * n_users
            purchase_probability = purchase_ratio * viewing_probability
            self._counts[current.id][1] += purchase_probability * n_users
            expected_purchase_count = self._config.purchase_amounts[class_][current.id].get_expectation()
            result_ = product_price * purchase_probability * n_users * expected_purchase_count
            result_ = round(result_, 2)  # 2 because we want cents :)
            first_p: Optional[ObservationProbability]
            second_p: Optional[ObservationProbability]
            # Calculation of the primary product done
            first_p, second_p = current.secondary_products[class_]
            new_primaries = clicked_primaries + (current.id,)
            if first_p is not None:
                # first_p[1] is the graph weight, first_p[0] is the product
                result_ += emulate_path(new_primaries, first_p[1] * purchase_probability * 1, first_p[0])
            if second_p is not None:
                # Now also Lambda has to be multiplied to the product because its a secondary product
                result_ += emulate_path(new_primaries, second_p[1] * purchase_probability * self._config.lambda_,
                                        second_p[0])
            return result_
        # Probability that the customer sees a given product depends on the alpha distribution
        reward = emulate_path((), self._environment.get_expected_alpha(class_)[product.id + 1], product)
        for prod in self._products:
            self._counts[prod.id][0] = round(self._counts[prod.id][0], 2)
            self._counts[prod.id][1] = round(self._counts[prod.id][1], 2)
        return round(reward, 2)

    def calculate_total_expected_reward(self, price_indexes: Tuple[int, ...]) -> Tuple[List[Reward], List[CR]]:
        rewards_result = []
        self._counts = {product.id: [0, 0] for product in self._products} # 0 is clicked, 1 is bought
        for product in self._products:
            rewards_result.append(0)
            for class_ in list(CustomerClass):
                rewards_result[-1] += self.calculate_reward_of_product(price_indexes[product.id], product, class_)
        cr_results = [self._counts[product.id][1] / self._counts[product.id][0] if self._counts[product.id] != 0 else 0 for
            product in self._products]
        return rewards_result, cr_results

    def _clairvoyant_reward_calculate(self, price_indexes) -> Tuple[List[Reward], List[CR]]:
        return self.calculate_total_expected_reward(price_indexes)

    def calculate_potential_candidate(self, pulled_arm: int):
        x = self.candidate_price_indexes
        new_price_indexes = x[:pulled_arm] + (x[pulled_arm] + 1,) + x[pulled_arm + 1:]
        if new_price_indexes[pulled_arm] == len(self._products[0].candidate_prices):
            return None
        self._environment.new_day()  # We have a new experiment, thus new day
        current_rewards, current_cr = self.calculate_total_expected_reward(new_price_indexes)

        if sum(current_cr) > self.current_cr:
            return sum(current_cr), new_price_indexes

    def _iterate_once(self) -> bool:
        """

        :return: Returns False if no more iterations are possible
        """
        ## By hw text, there are five trials for each iteration of the algorithm
        best_cr, best_price_index = max(
            (item for item in (self.calculate_potential_candidate(i) for i in range(len(self.candidate_price_indexes)))
             if item),
            key=itemgetter(0), default=(None, None))
        if best_cr is None:
            return False
        if best_cr < self.current_cr:
            return False
        self.current_cr, self.candidate_price_indexes = best_cr, best_price_index
        return True
