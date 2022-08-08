from dataclasses import dataclass, field
from operator import itemgetter
from typing import List

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
        for learner in self.learners:
            np_random.reset_seed()
            learner.refresh_vars(self.products, self.environment, self.config)
            self.environment.reset_day()
            learner.reset()
            learner.run_experiment(days, log=log, plot_graphs=plot_graphs, verbose=verbose)
