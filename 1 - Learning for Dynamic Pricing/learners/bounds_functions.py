# Tasks
## 1 - Average regret & reward computed over a significant number of runs 
## 2 - The ratio between the empirical regret and the upper bound

# import libraries
import numpy as np

## Compute the empirical regret
def expected_reward(Time_horizon, arms_rewards_list):
    # arms_rewards_list : list of rewards received at each time t
    return sum(arms_rewards_list)

# compute the average regret
def average_regret(clairvoyant_reward, Time_horizon, arms_rewards_list):
    # clairvoyant_reward : the reward of the clairvoyant learner
    # Time_horizon : the time horizon of the experiment
    # arms_pulled_list : the list of arms pulled by the learner we want to compute the average regret for
    # arms_rewards_list : the list of rewards of the arms pulled by the learner we want to compute the average regret for
    return Time_horizon * clairvoyant_reward - expected_reward(Time_horizon, arms_rewards_list)

# function that computes the upper bound of the UCB learner
def UCB_regret_UB(Time_horizon, clairvoyant_reward, arms_rewards_list):
		# clairvoyant reward : when we code it
		# arms_rewards_list : it is on get get_product_rewards
    delta_a = [4 * np.log(Time_horizon) / (clairvoyant_reward - arms_rewards_list[i])  
                + 8 * (clairvoyant_reward - arms_rewards_list[i])
                for i in range(len(arms_rewards_list))]
		# this is shit XD
    return sum(delta_a)


### function that computes the upper bound of the TS learner
# define the kullback leibler divergence
def KL_divergence(p, q):
		# p & q : probability distribution
    return np.sum(p * np.log(p / q))
		# it is in scipy 

def TS_regret_UB(Time_horizon, epsilon, clairvoyant_reward, C, arms_rewards_list):
		# epsilon & C values
    delta_a = [(np.log(Time_horizon) + np.log(np.log(Time_horizon))) 
                * (clairvoyant_reward - arms_rewards_list[i]) / KL_divergence(clairvoyant_reward, arms_rewards_list[i])
                for i in range(len(arms_rewards_list))]
    return (1 + epsilon) * sum(delta_a) + C


# function that computes the upper bound of the Gaussian-TS learner
def Gaussian_TS_regret_UB(Time_horizon, variance, N_arms, info_gain, delta):
    # what's the information gain in our case ????
		# We already have them ^^
    B = 8 * np.log(Time_horizon**4 * N_arms / (6*delta))
    bound = np.sqrt((8/np.log(1 + 1/variance**2)) * Time_horizon * B * info_gain)


# function that computes the upper bound of the Gaussian-TS learner
def Gaussian_TS_regret_UB(Time_horizon, variance, N_arms, info_gain, delta):
    # what's the information gain in our case ????
		# We already have them ^^
    B = 8 * np.log(Time_horizon**4 * N_arms / (6*delta))
    bound = np.sqrt((8/np.log(1 + 1/variance**2)) * Time_horizon * B * info_gain)