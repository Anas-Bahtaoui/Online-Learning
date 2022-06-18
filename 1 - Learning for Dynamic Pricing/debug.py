from typing import List
import preamble
from learners import step3, GaussianTSLearner, GreedyLearner, UCBLearner, Learner, step4, step5
from entities import Dirichlet, Simulation, CustomerTypeBased, SimulationConfig
from production import LAMBDA_, product_configs, purchase_amounts, customer_counts

secondary_product_professional: List[List[float]] = [[0] * 5 for _ in range(5)]
secondary_product_beginner_young = secondary_product_professional
secondary_product_beginner_old = secondary_product_professional

secondaries = CustomerTypeBased(
    professional=secondary_product_professional,
    young_beginner=secondary_product_beginner_young,
    old_beginner=secondary_product_beginner_old
)

dirichlet = Dirichlet([100, 100, 100, 100, 100, 100])
dirichlets: CustomerTypeBased[Dirichlet] = CustomerTypeBased(
    professional=dirichlet,
    young_beginner=dirichlet,
    old_beginner=dirichlet,
)

config = SimulationConfig(
    lambda_=LAMBDA_,
    product_configs=product_configs,
    secondaries=secondaries,
    purchase_amounts=purchase_amounts,
    customer_counts=customer_counts,
    dirichlets=dirichlets,
)

learners: List[Learner] = [
    GreedyLearner(),
    UCBLearner(step3),
    GaussianTSLearner(step3),
    UCBLearner(step4),
    GaussianTSLearner(step4),
    UCBLearner(step5),
    GaussianTSLearner(step5),
]
if __name__ == '__main__':
    simulation = Simulation(config, learners)
    simulation.run(50, log=False, plot_graphs=True, verbose=False)
