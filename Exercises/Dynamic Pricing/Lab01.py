import numpy as np
import matplotlib.pyplot as plt

class env:
        def __init__(self, probs, prices):
            self.prices = prices
            self.probs = probs
            
        def round(self, arm_pulled):
            conv = np.random.binomial(n=1, p=self.probs[arm_pulled])
            reward = conv * self.prices[arm_pulled]
            return reward

class learner:
        def __init__(self, n_arm):
            self.t = 0
            self.n_arm = n_arm
            self.rewards = []
            self.rewards_per_arm = [[] for _ in range(n_arm)]
            self.pulled = []
            
        def reset(self):
            self.__init__(self.n_arm, self.prices)
            
        def act(self):
            pass
        
        def update(self, arm_pulled, reward):
            self.t += 1
            self.rewards.append(reward)
            self.rewards_per_arm[arm_pulled].append(reward)
            self.pulled.append(arm_pulled)
            
class ucb(learner):
    def __init__(self, n_arm, prices):
        super().__init__(n_arm)
        #self.T = T
        self.mean = np.array([0]*n_arm)
        self.widths = np.array([np.inf for _ in range(n_arm)])
        self.prices = prices
        
    def act(self):
        idx = np.argmax((self.mean + self.widths)*self.prices)
        return idx
    
    def update(self, arm_pulled, reward):
        super().update(arm_pulled, reward)
        self.mean[arm_pulled] = np.mean(self.rewards_per_arm[arm_pulled])
        for idx in range(self.n_arm):
            n = len(self.rewards_per_arm[idx])
            if n>0:
                self.widths[idx] = np.sqrt(2*np.max(self.prices)*np.log(self.t)/n)
            else:
                self.widths[idx] = np.inf
                
p = [0.5, 0.1, 0.2, 0.9]
prices = [100, 400, 600, 60]

pricing_env = env(p, prices)
ag1 = ucb(len(p),prices)
T = 1000
opt = np.max([a*b for a,b in zip(p,prices)] )
N_exp = 500

R = []
for _ in range(N_exp):
    instant_regret = []
    ag1.reset()
    for t in range(T):
        pulled_arm = ag1.act()
        rew = pricing_env.round(pulled_arm)
        ag1.update(pulled_arm, rew)
        instant_regret.append(opt - rew)
    #cumulative_regret = np.append(cumulative_regret, instant_regret)
    cumulative_regret = np.cumsum(instant_regret)
    R.append(cumulative_regret)
mean_R = np.mean(R, axis=0)
std_dev = np.std(R, axis=0)/np.sqrt(N_exp)

plt.plot(mean_R)
plt.fill_between(range(T), mean_R-std_dev, mean_R+std_dev, alpha = 0.4)
plt.show()