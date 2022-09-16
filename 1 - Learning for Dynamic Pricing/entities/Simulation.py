from dataclasses import dataclass, field
from operator import itemgetter
from typing import List, Callable

from tqdm import tqdm

from Environment import Environment
from Product import Product
from basic_types import SimulationConfig, CustomerClass
from random_ import np_random


@dataclass
class Simulation:
    config: SimulationConfig
    learners: List['Learner']
    products: List[Product] = field(init=False)
    environment: Environment = field(init=False)

    def __post_init__(self):

        # Create products
        product_configs = self.config.product_configs
        secondaries = self.config.secondaries
        products = [Product(id_, *product_config) for id_, product_config in enumerate(product_configs)]
        for class_ in CustomerClass:
            for from_, targets in enumerate(secondaries[class_]):
                first, second = sorted(enumerate(targets), key=itemgetter(1), reverse=True)[:2]
                if first[1] == 0:
                    pass
                elif second[1] == 0:
                    products[from_].add_secondary_products(class_, products[first[0]], first[1])
                else:
                    products[from_].add_secondary_products(class_, products[first[0]], first[1], products[second[0]],
                                                           second[1])
        self.products = products

        # Create environment
        self.environment = Environment(self.config.dirichlets)

    def run(self, days: int, *, log: bool, plot_graphs: bool, verbose: bool):
        clairvoyant_reward = 0
        for learner in self.learners:
            np_random.reset_seed()
            learner.refresh_vars(self.products, self.environment, self.config)
            self.environment.reset_day()
            learner.reset()
            learner.run_experiment(days, log=log, plot_graphs=plot_graphs, verbose=verbose)
            if not clairvoyant_reward and hasattr(learner, "_new_day"):
                clairvoyant_reward, best_indexes = self.run_clairvoyant(lambda x: learner._new_day(x, False))
        for learner in self.learners:
            learner.clairvoyant = clairvoyant_reward
        if log:
            print(f"Clairvoyant: {clairvoyant_reward}")
            print(f"on product indexes: {best_indexes}")
    def run_clairvoyant(self, calculate: Callable[[List[int]], List[float]]):
        from itertools import product
        product_count = len(self.products)
        price_index_count = len(self.products[0].candidate_prices)
        max_reward = 0
        best_indexes = ()
        all_price_indexes = list(product(range(price_index_count), repeat=product_count))
        with tqdm(total=len(all_price_indexes), leave=False) as pbar:
            pbar.set_description(f"Clairvoyant")
            for price_indexes in all_price_indexes[:1]:
                total_reward = sum(calculate(list(price_indexes)))
                if total_reward > max_reward:
                    max_reward = total_reward
                    best_indexes = price_indexes
                pbar.update(1)
        return max_reward, best_indexes
