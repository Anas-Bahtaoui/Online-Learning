from typing import List

from learners.GaussianThompsonLearner import GaussianTSLearner
from learners.GreedyLearner import GreedyLearner
from learners.Learner import Learner
from learners.UCBLearner import UCBLearner
from entities import ProductConfig, SimulationConfig, CustomerTypeBased, Simulation, Dirichlet, PositiveIntegerGaussian as PIG
from learners.BanditLearner import step3, step4, step5, step6_sliding_window, step6_change_detection, step7

LAMBDA_ = 0.1

product_configs: List[ProductConfig] = [
    ProductConfig("T-shirt", [3, 12, 20, 40]),
    ProductConfig("Shorts", [5, 13, 22, 35]),
    ProductConfig("Towel", [2, 10, 15, 20]),
    ProductConfig("Dumbbells", [5, 16, 34, 70]),
    ProductConfig("Protein Powder", [15, 18, 25, 35]),

]
# @formatter:off
secondary_product_professional: List[List[float]] = [
    [0,    0,    0.4,  0,    0],
    [0.3,  0,    0,    0.1,  0],
    [0,    0.25, 0,    0.05, 0],
    [0.15, 0,    0,    0,    0],
    [0,    0,    0,    0,    0],
]
# TODO: Custom values for these
secondary_product_beginner_young = secondary_product_professional
secondary_product_beginner_old = secondary_product_professional
# @formatter:on

secondaries = CustomerTypeBased(
    professional=secondary_product_professional,
    young_beginner=secondary_product_beginner_young,
    old_beginner=secondary_product_beginner_old
)

# TODO: is the Integer Gaussian good for this
# TODO: After we determine our products, update these values.

purchase_amounts: CustomerTypeBased[List[PIG]] = CustomerTypeBased(
    professional=(PIG(5, 1), PIG(1, 1), PIG(3, 1), PIG(1, 1), PIG(1, 1)),
    young_beginner=(PIG(8, 1), PIG(2, 1), PIG(6, 1), PIG(1, 1), PIG(2, 1)),
    old_beginner=(PIG(15, 2), PIG(4, 1), PIG(8, 1), PIG(2, 1), PIG(8, 1)),
)

customer_counts: CustomerTypeBased[PIG] = CustomerTypeBased(
    professional=PIG(mean=50, variance=4),
    young_beginner=PIG(mean=100, variance=6),
    old_beginner=PIG(mean=30, variance=5),
)
dirichlets: CustomerTypeBased[Dirichlet] = CustomerTypeBased(
    professional=Dirichlet([100, 100, 100, 100, 100, 100]),
    young_beginner=Dirichlet([100, 100, 100, 100, 100, 100]),
    old_beginner=Dirichlet([100, 100, 100, 100, 100, 100]),
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
    GreedyLearner()
]

for step in [step3, step4, step5, step6_sliding_window, step6_change_detection, step7]:
    for Learner in [GaussianTSLearner, UCBLearner]:
        learners.append(Learner(step))

if __name__ == '__main__':
    simulation = Simulation(config, learners)
    simulation.run(50, log=False, plot_graphs=True)
