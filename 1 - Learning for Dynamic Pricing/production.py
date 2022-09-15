import preamble

from typing import List

from SlidingUCBLearner import SlidingUCBLearner
from entities import ProductConfig, SimulationConfig, CustomerTypeBased, Simulation, Dirichlet, \
    PositiveIntegerGaussian as PIG, Constant
from learners import *

LAMBDA_ = 0.85

product_configs: List[ProductConfig] = [
    ProductConfig("T-shirt", [3, 20, 50, 96]),  # Also here.
    ProductConfig("Shorts", [10, 25, 70, 90]),
    ProductConfig("Towel", [1, 3, 5, 7]),
    ProductConfig("Dumbbells", [5, 10, 15, 30]),
    # TODO: The previous price created an unstability, this one makes uncertainty
    ProductConfig("Protein Powder", [15, 30, 60, 80]),

]
# @formatter:off
# Rows: from
# Columns: to
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

purchase_amounts: CustomerTypeBased[List[PIG]] = CustomerTypeBased(
    # TODO: Values for how many are bought
    professional=(Constant(2), Constant(1), Constant(3), Constant(3), Constant(1)),
    young_beginner=(Constant(1), Constant(2), Constant(6), Constant(1), Constant(2)),
    old_beginner=(Constant(2), Constant(2), Constant(8), Constant(2), Constant(3)),
)

# TODO: Assign ratios based on one number, so we can control convergence
customer_counts: CustomerTypeBased[PIG] = CustomerTypeBased(
    # TODO: Values here
    professional=Constant(50),
    young_beginner=Constant(100),
    old_beginner=Constant(30),
)
dirichlets: CustomerTypeBased[Dirichlet] = CustomerTypeBased(
    # TODO: We want class specific values
    professional=Dirichlet([100, 90,  110, 200, 200, 90]),
    young_beginner=Dirichlet([100, 110,  130, 190, 180, 120]),
    old_beginner=Dirichlet([100, 100,  140, 210, 190, 100]),
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

for step in [
    step3,
    # step4,
    # step5,
    step6_sliding_window,
    step6_change_detection,
]:#step6_sliding_window, step6_change_detection, step7]:
    for Learner in [UCBLearner, NewerGTSLearner]:
        learners.append(Learner(step))

learners.append(SlidingUCBLearner(step6_sliding_window._replace(non_stationary=None)))

RUN_COUNT = 50
if __name__ == '__main__':
    simulation = Simulation(config, learners)
    simulation.run(RUN_COUNT, plot_graphs=True)
