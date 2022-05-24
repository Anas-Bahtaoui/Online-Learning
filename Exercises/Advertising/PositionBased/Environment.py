from typing import List

import numpy as np
from dataclasses import dataclass


@dataclass
class EnvironmentPBM:
    n_arms: int
    n_positions: int
    arm_probabilities: np.ndarray
    position_probabilities: np.ndarray

    def __post_init__(self):
        assert self.n_arms == len(self.arm_probabilities)
        assert self.n_positions == len(self.position_probabilities)

    def round(self, pulled_super_arm):
        assert len(pulled_super_arm) == len(self.position_probabilities)
        position_obs = np.random.binomial(1, self.position_probabilities)
        arm_obs = np.random.binomial(1, self.arm_probabilities[pulled_super_arm])
        return arm_obs * position_obs
