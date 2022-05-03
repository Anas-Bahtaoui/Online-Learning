from typing import List

from BanditLearner import step3
from Environment import constant_generator
from GreedyLearner import GreedyLearner
from Learner import Learner
from TSLearner import TSLearner
from UCBLearner import UCBLearner
from sample import generate_sample_greedy_environment

if __name__ == '__main__':
    env = generate_sample_greedy_environment()
    env.alpha_generator = constant_generator
    learners: List[Learner] = [GreedyLearner(env), UCBLearner(env, step3), TSLearner(env, step3)]
    for learner in learners:
        learner.run_experiment(env, 100)

# TODO: Tune down the model and debug whatever is happening with the bandit algorithms
