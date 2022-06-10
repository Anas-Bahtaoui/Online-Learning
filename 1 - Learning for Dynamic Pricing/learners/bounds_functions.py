'''
For the Steps 3-7, in the algorithm evaluation, report:
## the average regret and reward computed over a significant number of runs,their standard deviation, 
## when theoretical bounds are available, also report the ratio between the empiric regret 
#                                                                       and the upper bound.
'''

# import libraries
import numpy as np

## Compute the expected reward : 
## definition (on the slides) as the sum of the rewards we get at each time step
def expected_reward(arms_rewards_list):
    # arms_rewards_list : list of rewards received at each time t
    return sum(arms_rewards_list)

# compute the average regret :
## definition (on the slides) difference between claivoyant reward and the expected reward of our learner
def average_regret(clairvoyant_reward, arms_rewards_list):
    # claivoyant reward : reward we get from the clairvoyant learner
    # arms_rewards_list : list of rewards received at each time t
    Time_horizon = len(arms_rewards_list)
    return Time_horizon * clairvoyant_reward - expected_reward(arms_rewards_list)

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