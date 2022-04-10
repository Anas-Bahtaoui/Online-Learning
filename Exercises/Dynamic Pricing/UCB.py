import numpy as np

class env:
        def __init__(self, probs):
            self.probs
            
        def  round(self, arm_pulled):
            reward = np.random.binomial(n=1, p=self.probs[arm_pulled])
            return reward

class learner:
        def __init__(self, n_arm):
            self.t = 0
            self.n_arm = n_arm
            self.rewards = []
            self.rewards_per_arm = [[] for _ in range(n_arm)]
            
        def reset(self):
            self.__init__(self.n_arm)
            
        def act(self):
            pass
        def update(self, arm_pulled, reward):
            self.rewards.append(reward)
            self.rewards_per_arm[arm_pulled].append(reward)
            
class ucb(learner):
    def __init__(self, n_arm):
        super().__init__(n_arm)
        self.mean = np.array(n_arm)
        self.widths = np.array([np.inf for _ in range(n_arm)])
        
    def act(self):
        idx = np.argmax(self.mean + self.widths)
        return idx
    
    def update(self, arm_pulled, reward):
        super().update(arm_pulled, reward)
        self.mean = np.mean(self.rewards_per_arm[arm_pulled])
        for idx in range(self.n_arm):
            n = len(self.rewards_per_arm[idx])
            if n>0:
                self.widths[idx] = np.sqrt(2*np.log(self.t))
            else:
                self.widths[idx] = np.inf