import numpy as np

SEED = 42


class RandomGenerator:
    def __init__(self):
        self._np_random = np.random.default_rng(SEED)

    def reset_seed(self):
        self._np_random = np.random.default_rng(SEED)

    def __getattr__(self, item):
        return getattr(self._np_random, item)


np_random = RandomGenerator()
