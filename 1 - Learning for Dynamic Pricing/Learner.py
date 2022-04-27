"""
    Basic Learner.
"""
from operator import itemgetter
from typing import Tuple, Annotated, Optional

from matplotlib import pyplot as plt
from sklearn.tree import plot_tree
from Environment import Environment, constant_generator
from Customer import CustomerClass, purchase_amounts, customer_counts, reservation_price_distribution_from_curves
from Product import Product, ObservationProbability
from sample import generate_sample_greedy_environment


import numpy as np


class Learner:
    """
        This is the same Learner of the prof
    """

    def __init__(self, n_arms):
        """
        :param n_arms:
        """
        self.n_arms = n_arms

        self.t = 0
        self.rewards_per_arm = [[] for _ in range(n_arms)]
        self.collected_rewards = np.array([])

    def update_observations(self, pulled_arm, reward):
        """
        :param pulled_arm:
        :param reward:
        :return:
        """
        self.rewards_per_arm[pulled_arm].append(reward)
        self.collected_rewards = np.append(self.collected_rewards, reward)


class GreedyLearner:
    def __init__(self, environment: Environment):
        self.candidate_price_indexes = (0, 0, 0, 0, 0)
        self.current_reward = 0
        self.env = environment

    def calculate_reward_of_product(self, price_index: int, product: Product, class_: CustomerClass) -> float:
        current_price_indexes = self.candidate_price_indexes[:product.id] + (
            price_index,) + self.candidate_price_indexes[product.id + 1:]
        n_users = customer_counts[class_].get_expectation()

        def emulate_path(clicked_primaries: Tuple[int, ...], viewing_probability: float, current: Product):
            if current.id in clicked_primaries:
                return 0

            product_price = product.candidate_prices[current_price_indexes[product.id]]
            reservation_price = reservation_price_distribution_from_curves(class_, product.id, product_price).get_expectation()
            is_purchased = reservation_price >= product_price
            if not is_purchased:
                return 0
            expected_purchase_count = purchase_amounts[class_][product.id].get_expectation()
            result_ = (product_price - product.production_cost) * viewing_probability * n_users * expected_purchase_count
            result_ = round(result_, 2) # 2 because we want cents :)
            first_p: Optional[ObservationProbability]
            second_p: Optional[ObservationProbability]
            first_p, second_p = product.secondary_products
            new_primaries = clicked_primaries + (current.id,)
            if first_p is not None:
                result_ += emulate_path(new_primaries, first_p[1] * viewing_probability * 1, first_p[0])
            if second_p is not None:
                result_ += emulate_path(new_primaries, second_p[1] * viewing_probability * self.env.lambda_, second_p[0])
            return result_

        return emulate_path((), self.env.alpha[product.id + 1], product)

    def calculate_total_expected_reward(self, price_indexes: Tuple[int, ...]) -> float:
        result = 0
        for product in self.env.products:
            for class_ in list(CustomerClass):
                result += self.calculate_reward_of_product(price_indexes[product.id], product, class_)
        return result

    def calculate_potential_candidate(self, pulled_arm: int):
        x = self.candidate_price_indexes
        new_price_indexes = x[:pulled_arm] + (x[pulled_arm] + 1,) + x[pulled_arm + 1:]
        if new_price_indexes[pulled_arm] == len(self.env.products[0].candidate_prices):
            return None
        print("New experiment on day", self.env.day, "with price indexes", new_price_indexes)
        self.env.new_day()  # We have a new experiment, thus new day
        reward = self.calculate_total_expected_reward(new_price_indexes)
        if reward > self.current_reward:
            return reward, new_price_indexes

    def iterate_once(self) -> bool:
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


if __name__ == '__main__':
    env = generate_sample_greedy_environment()
    env.alpha_generator = constant_generator
    learner = GreedyLearner(env)
    running = True
    cnt = 0
    
    currentReward = []
    currentCNT = []
    p1 = []
    p2 = []
    p3 = []
    p4 = []
    p5 = []
    
    while running:
        running = learner.iterate_once()
        print(f"iteration {cnt}:")
        print("Indexes", learner.candidate_price_indexes)
        print("Reward", learner.current_reward)
        cnt += 1
        
        currentCNT.append(cnt)
        # Save the current reward
        currentReward.append(learner.current_reward)
        # Store the price indexes
        p1.append(learner.candidate_price_indexes[0])
        p2.append(learner.candidate_price_indexes[1])
        p3.append(learner.candidate_price_indexes[2])
        p4.append(learner.candidate_price_indexes[3])
        p5.append(learner.candidate_price_indexes[4])        
                
    print("###############################################\n")
    print("Done!")
    print("Identified price indexes:", learner.candidate_price_indexes)
    
    # Plot the current_reward over iterations    
    plt.plot(currentCNT, currentReward)
    plt.xlabel("Iteration")
    plt.ylabel("Reward")
    plt.title("Greedy Algorithm Reward")
    plt.show()
    
    # Plot the prices p1, p2, p3, p4 and p5 over the iterations
    priceP1 = []
    priceP2 = []
    priceP3 = []
    priceP4 = []
    pricep5 = []
    
    for i in range(len(currentCNT)):
        priceP1.append(env.products[0].candidate_prices[p1[i]])
        priceP2.append(env.products[1].candidate_prices[p2[i]])
        priceP3.append(env.products[2].candidate_prices[p3[i]])
        priceP4.append(env.products[3].candidate_prices[p4[i]])
        pricep5.append(env.products[4].candidate_prices[p5[i]])
                
    plt.plot(currentCNT, priceP1, label="p1")
    plt.plot(currentCNT, priceP2, label="p2")
    plt.plot(currentCNT, priceP3, label="p3")
    plt.plot(currentCNT, priceP4, label="p4")
    plt.plot(currentCNT, pricep5, label="p5")
    
    plt.xlabel("Iteration")
    plt.ylabel("Prices per product")
    plt.title("Greedy Algorithm Prices")
    plt.legend()
    plt.show()    