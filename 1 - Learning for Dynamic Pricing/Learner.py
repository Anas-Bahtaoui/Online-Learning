from collections import defaultdict
from typing import List, Tuple
from matplotlib import pyplot as plt

from parameters import products

ShallContinue = bool
Reward = float
PriceIndexes = List[int]


class Learner:
    name: str

    def iterate_once(self) -> Tuple[ShallContinue, Reward, PriceIndexes]:
        raise NotImplementedError()

    def run_experiment(self, max_days: int, *, log: bool = True, plot_graphs: bool = True) -> None:
        running = True
        cnt = 0

        rewards = []
        products_ = defaultdict(list)
        while running and cnt < max_days:
            running, current_reward, candidate_price_indexes = self.iterate_once()
            if log:
                print(f"iteration {cnt}:")
                print("Indexes", candidate_price_indexes)
                print("Reward", current_reward)
            cnt += 1

            # Save the current reward
            rewards.append(current_reward)
            # Store the price indexes
            for product_i in range(5):
                products_[product_i].append(candidate_price_indexes[product_i])
        if log:
            print("###############################################\n")
            print("Done!")
            print("Identified price indexes:", candidate_price_indexes)

        if plot_graphs:
            x_iteration = list(range(1, cnt + 1))
            # Plot the current_reward over iterations
            plt.plot(x_iteration, rewards)
            plt.xlabel("Iteration")
            plt.ylabel("Reward")
            plt.title(f"{self.name} Reward")
            plt.show()

            # Plot the prices p1, p2, p3, p4 and p5 over the iterations
            prices = defaultdict(list)
            for productId in range(5):
                for i in range(cnt):
                    prices[productId].append(products[productId].candidate_prices[products_[productId][i]])

                plt.plot(x_iteration, prices[productId], label=products[productId].name)
            plt.xlabel("Iteration")
            plt.ylabel("Prices per product")
            plt.title(f"{self.name} Prices")
            plt.legend()
            plt.show()

