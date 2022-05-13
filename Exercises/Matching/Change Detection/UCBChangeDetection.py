# To detect changes we will use the cumulative sum

import numpy as np
from project.Exercises.DynamicPricing.ThompsonSampling.NonStationaryEnvironment import NonStationaryEnvironment
import matplotlib.pyplot as plt
from UCBMatching import UCBMatching
from scipy.optimize import linear_sum_assignment


class CumSum:
    def __init__(self, M, eps, h):
        self.M = M
        self.eps = eps
        self.h = h
        self.t = 0
        self.reference = 0
        self.g_plus = 0
        self.g_minus = 0

    def has_changed(self, sample) -> bool:
        self.t += 1
        if self.t <= self.M:
            self.reference += sample / self.M
            return False
        else:
            s_plus = (sample - self.reference) - self.eps
            s_minus = -(sample - self.reference) - self.eps
            self.g_plus = max(0, self.g_plus + s_plus)
            self.g_minus = max(0, self.g_minus + s_minus)
            return self.g_plus > self.h or self.g_minus > self.h

    def reset(self):
        self.t = 0
        # Because we call reset after one change has been detected.
        # So we don't want to lose the reference. but why? Isn't it better to have a new reference to start afresh?
        # TODO: original code doesn't have this line
        # self.reference = 0
        self.g_plus = 0
        self.g_minus = 0


class UCBChangeDetection(UCBMatching):
    def __init__(self, n_arms, n_rows, n_cols, eps=0.05, h=20, M=100, alpha=0.01):
        super().__init__(n_arms, n_rows, n_cols)
        self.change_detectors = [CumSum(M, eps, h) for _ in range(self.n_arms)]
        self.valid_rewards_per_arms = [[] for _ in range(self.n_arms)]
        self.detections = [[] for _ in range(n_arms)]
        self.alpha = alpha

    def pull_arm(self) -> (int, int):
        if np.random.binomial(1, 1 - self.alpha):
            upper_conf = self.empirical_means + self.confidence
            upper_conf[np.isinf(upper_conf)] = 1e3
            return linear_sum_assignment(-upper_conf.reshape(self.n_rows, self.n_cols))
        else:
            costs_random = np.random.randint(0, 10, size=(self.n_rows, self.n_cols))
            return linear_sum_assignment(costs_random)

    def update(self, pulled_arms, reward):
        self.t += 1
        pulled_arm_flat = np.ravel_multi_index(pulled_arms, (self.n_rows, self.n_cols))
        for pulled_arm, reward in zip(pulled_arm_flat, reward):
            if self.change_detectors[pulled_arm].has_changed(reward):
                self.detections[pulled_arm].append(self.t)
                self.valid_rewards_per_arms[pulled_arm] = []
                self.change_detectors[pulled_arm].reset()

            self.update_observations(pulled_arm, reward)
            self.empirical_means[pulled_arm] = np.mean(self.valid_rewards_per_arms[pulled_arm])

        total_valid_samples = sum([len(self.valid_rewards_per_arms[a]) for a in range(self.n_arms)])
        for a in range(self.n_arms):
            n_samples = len(self.valid_rewards_per_arms[a])
            self.confidence[a] = np.sqrt(2 * np.log(total_valid_samples) / n_samples) if n_samples > 0 else np.inf

    def update_observations(self, pulled_arm, reward):
        self.rewards_per_arm[pulled_arm].append(reward)
        self.valid_rewards_per_arms[pulled_arm].append(reward)
        self.collected_rewards = np.append(self.collected_rewards, reward)


if __name__ == '__main__':
    p0 = np.array([[1 / 4, 1, 1 / 4], [1 / 2, 1 / 4, 1 / 4], [1 / 4, 1 / 4, 1]])
    p1 = np.array([[1, 1 / 4, 1 / 4], [1 / 2, 1 / 4, 1 / 4], [1 / 4, 1 / 4, 1]])
    p2 = np.array([[1, 1 / 4, 1 / 4], [1 / 2, 1, 1 / 4], [1 / 4, 1 / 4, 1]])
    P = [p0, p1, p2]
    T = 3000
    n_exp = 10
    regret_cusum = np.zeros((n_exp, T))
    regret_ucb = np.zeros((n_exp, T))
    detections = [[] for _ in range(n_exp)]
    M = 100
    eps = 0.1
    h = np.log(T) * 2
    for j in range(n_exp):
        e_UCB = NonStationaryEnvironment(p0.size, P, T)
        e_CD = NonStationaryEnvironment(p0.size, P, T)
        learner_CD = UCBChangeDetection(p0.size, *p0.shape, M, eps, h)
        learner_UCB = UCBMatching(p0.size, *p0.shape)
        opt_rew = []
        rew_CD = []
        rew_UCB = []
        for t in range(T):
            p = P[int(t / e_UCB.phase_size)]
            opt = linear_sum_assignment(-p)
            opt_rew.append(p[opt].sum())
            pulled_arm = learner_CD.pull_arm()
            reward = e_CD.round(pulled_arm)
            learner_CD.update(pulled_arm, reward)
            rew_CD.append(reward.sum())
            pulled_arm = learner_UCB.pull_arm()
            reward = e_UCB.round(pulled_arm)
            learner_UCB.update(pulled_arm, reward)
            rew_UCB.append(reward.sum())
        regret_cusum[j, :] = np.cumsum(opt_rew) - np.cumsum(rew_CD)
        regret_ucb[j, :] = np.cumsum(opt_rew) - np.cumsum(rew_UCB)
    plt.figure(0)
    plt.ylabel('Regret')
    plt.xlabel('t')
    plt.plot(np.mean(regret_cusum, axis=0))
    plt.plot(np.mean(regret_ucb, axis=0))
    plt.legend(['CD-UCB', 'UCB'])
    plt.show()
