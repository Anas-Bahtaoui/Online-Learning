from UCB import UCB
import numpy as np
from scipy.optimize import linear_sum_assignment


class UCBMatching(UCB):
    def __init__(self, n_arms, n_rows, n_cols):
        if n_rows * n_cols != n_arms:
            raise ValueError(
                f"Number of arms {n_arms} must be equal to number of rows {n_rows} * number of columns {n_cols}")
        super().__init__(n_arms)
        self.n_rows = n_rows
        self.n_cols = n_cols

    def pull_arm(self) -> (int, int):
        upper_conf = self.empirical_means + self.confidence
        upper_conf[np.isinf(upper_conf)] = 1e3
        return linear_sum_assignment(-upper_conf.reshape(self.n_rows, self.n_cols))

    def update(self, pulled_arms, reward):
        self.t += 1
        # We need to pull multiple arms
        pulled_arm_flat = np.ravel_multi_index(pulled_arms, (self.n_rows, self.n_cols))
        for pulled_arm, reward in zip(pulled_arm_flat, reward):
            self.update_observations(pulled_arm, reward)
            self.empirical_means[pulled_arm] = (self.empirical_means[pulled_arm] * (self.t - 1) + reward) / self.t

        for a in range(self.n_arms):
            n_samples = len(self.rewards_per_arm[a])
            self.confidence[a] = np.sqrt(2 * np.log(self.t) / n_samples) if n_samples > 0 else np.inf


if __name__ == '__main__':
    from project.Exercises.DynamicPricing.ThompsonSampling.Environment import Environment
    import matplotlib.pyplot as plt

    p = np.array([[1 / 4, 1, 1 / 4], [1 / 2, 1 / 4, 1 / 4], [1 / 4, 1 / 4, 1]])
    opt = linear_sum_assignment(-p)
    n_exp = 10
    T = 3000
    regret_ucb = np.zeros((n_exp, T))
    for e in range(n_exp):
        learner = UCBMatching(p.size, *p.shape)
        print(e)
        rew_UCB = []
        opt_rew = []
        env = Environment(p.size, p)
        for t in range(T):
            pulled_arm = learner.pull_arm()
            reward = env.round(pulled_arm)
            learner.update(pulled_arm, reward)
            rew_UCB.append(reward.sum())
            opt_rew.append(p[opt].sum())
        regret_ucb[e, :] = np.cumsum(opt_rew) - np.cumsum(rew_UCB)

    plt.figure(0)
    plt.plot(regret_ucb.mean(axis=0))
    plt.ylabel("Regret")
    plt.xlabel("Time")
    plt.show()
