from operator import itemgetter
from typing import Tuple, Optional

from Customer import CustomerClass, reservation_price_distribution_from_curves
from Learner import Learner, ShallContinue, Reward, PriceIndexes
from Product import Product, ObservationProbability
from parameters import environment, LAMBDA_, products, purchase_amounts, customer_counts


class GreedyLearner(Learner):
    name = "Greedy Algorithm"

    def iterate_once(self) -> Tuple[ShallContinue, Reward, PriceIndexes]:
        return self._iterate_once(), self.current_reward, list(self.candidate_price_indexes)

    def __init__(self):
        self.candidate_price_indexes = (0, 0, 0, 0, 0)
        self.current_reward = 0

    def calculate_reward_of_product(self, price_index: int, product: Product, class_: CustomerClass) -> float:
        current_price_indexes = self.candidate_price_indexes[:product.id] + (
            price_index,) + self.candidate_price_indexes[product.id + 1:]
        n_users = customer_counts[class_].get_expectation()

        def emulate_path(clicked_primaries: Tuple[int, ...], viewing_probability: float, current: Product):
            if current.id in clicked_primaries:
                return 0

            product_price = product.candidate_prices[current_price_indexes[product.id]]
            reservation_price = reservation_price_distribution_from_curves(class_, product.id,
                                                                           product_price).get_expectation()
            is_purchased = reservation_price >= product_price
            if not is_purchased:
                return 0
            expected_purchase_count = purchase_amounts[class_][product.id].get_expectation()
            # TODO: When I added purchase amounts, all went wrong.
            result_ = (
                              product_price - product.production_cost) * viewing_probability * n_users# * expected_purchase_count
            # TODO: Shall we also ignore counts, if we are ignoring them in the bandits?
            result_ = round(result_, 2)  # 2 because we want cents :)
            first_p: Optional[ObservationProbability]
            second_p: Optional[ObservationProbability]
            first_p, second_p = product.secondary_products
            new_primaries = clicked_primaries + (current.id,)
            if first_p is not None:
                result_ += emulate_path(new_primaries, first_p[1] * viewing_probability * 1, first_p[0])
            if second_p is not None:
                result_ += emulate_path(new_primaries, second_p[1] * viewing_probability * LAMBDA_,
                                        second_p[0])
            return result_

        return emulate_path((), environment.alpha[product.id + 1], product)

    def calculate_total_expected_reward(self, price_indexes: Tuple[int, ...]) -> float:
        result = 0
        for product in products:
            for class_ in list(CustomerClass):
                result += self.calculate_reward_of_product(price_indexes[product.id], product, class_)
        return result

    def calculate_potential_candidate(self, pulled_arm: int):
        x = self.candidate_price_indexes
        new_price_indexes = x[:pulled_arm] + (x[pulled_arm] + 1,) + x[pulled_arm + 1:]
        if new_price_indexes[pulled_arm] == len(products[0].candidate_prices):
            return None
        print("New experiment on day", environment.day, "with price indexes", new_price_indexes)
        environment.new_day()  # We have a new experiment, thus new day
        reward = self.calculate_total_expected_reward(new_price_indexes)
        print("Reward is ", reward)
        if reward > self.current_reward:
            return reward, new_price_indexes

    def _iterate_once(self) -> bool:
        """

        :return: Returns False if no more iterations are possible
        """
        best_reward, best_price_index = max(
            (item for item in (self.calculate_potential_candidate(i) for i in range(len(self.candidate_price_indexes)))
             if item),
            key=itemgetter(0), default=(None, None))
        if best_reward is None:
            return False
        if best_reward < self.current_reward:
            return False
        print("Better reward found", best_reward, "with price indexes", best_price_index)
        self.current_reward, self.candidate_price_indexes = best_reward, best_price_index
        return True

