from typing import List

from BanditLearner import step3, step4, step5, step6_sliding_window, step6_change_detection, step7
from GreedyLearner import GreedyLearner
from Learner import Learner
from UCBLearner import UCBLearner
from GaussianThompsonLearner import GaussianTSLearner
from parameters import environment

if __name__ == '__main__':

    learners: List[Learner] = [
        GreedyLearner(),
        # TODO: Remove after this line and use the lines below when all is done
        # Better looking sexy code :)

        UCBLearner(step3),
        GaussianTSLearner(step3)
    ]
    # for step in [step3, step4, step5, step6_sliding_window, step6_change_detection, step7]
    #     for Learner in [GaussianTSLearner, UCBLearner]:
    #         learners.append(Learner(step))

    for learner in learners:
        environment.reset_day()
        learner.run_experiment(50, log=False, plot_graphs=True)

# TODO: Tune down the model and debug whatever is happening with the bandit algorithms
