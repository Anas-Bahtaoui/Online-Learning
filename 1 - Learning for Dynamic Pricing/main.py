from typing import List

from BanditLearner import step3, step4, step5, step6_sliding_window, step6_change_detection, step7
from GreedyLearner import GreedyLearner
from Learner import Learner
from Simulation import Simulation
from UCBLearner import UCBLearner
from GaussianThompsonLearner import GaussianTSLearner
from production import config

if __name__ == '__main__':
    learners: List[Learner] = [
        GreedyLearner(),
        # TODO: Remove after this line and use the lines below when all is done
        # Better looking sexy code :)

        UCBLearner(step3),
        GaussianTSLearner(step3)
    ]


    Simulation(config, learners).run(50, log=True, plot_graphs=True)
# TODO: Tune down the model and debug whatever is happening with the bandit algorithms
