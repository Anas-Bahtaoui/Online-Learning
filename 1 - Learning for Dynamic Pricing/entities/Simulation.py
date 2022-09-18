from dataclasses import dataclass, field
from operator import itemgetter
from typing import List, Tuple, Dict

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
    experiments: Dict[str, List[Tuple[float, List['ExperimentHistoryItem']]]] = field(init=False)

    # Structure: learner_name -> experiment_index -> (absolute_clairvoyant, day_index -> (experiment_history))
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
        self.experiments = {}

    def run(self, days: int, experiment_count: int = 1, *, plot_graphs: bool):
        for learner in self.learners:
            np_random.reset_seed()
            n_experiment = experiment_count
            if "Greedy" in learner.name:
                n_experiment = 1  # Greedy is deterministic
            self.experiments[learner.name] = []
            for experiment_index in range(n_experiment):
                learner.refresh_vars(self.products, self.environment, self.config)
                self.environment.reset_day()
                learner.reset()
                learner.absolute_clairvoyant, learner.clairvoyant_indexes = self.run_clairvoyant(learner)
                learner.run_experiment(days, plot_graphs=plot_graphs, current_n=experiment_index + 1)
                self.experiments[learner.name].append((learner.absolute_clairvoyant, learner._experiment_history))

    def run_clairvoyant(self, learner: "Learner"):
        calculate = learner._clairvoyant_reward_calculate
        from itertools import product
        product_count = len(self.products)
        price_index_count = len(self.products[0].candidate_prices)
        max_reward = 0
        best_indexes = ()
        all_price_indexes = list(product(range(price_index_count), repeat=product_count))[:1]
        with tqdm(total=len(all_price_indexes), leave=False) as pbar:
            pbar.set_description(f"Clairvoyant")
            for price_indexes in all_price_indexes[:1]:
                total_reward = calculate(list(price_indexes))
                if total_reward > max_reward:
                    max_reward = total_reward
                    best_indexes = price_indexes
                pbar.update(1)
        return max_reward, best_indexes
