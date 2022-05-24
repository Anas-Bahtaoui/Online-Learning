"""
Definition of the Environment class. It is used to iterate over the days of the simulation.
    
Each day,there is a random number of potential customers. 
Each product i is associated with a probability alpha_i, which is the ratio of customers landing on the webpage in which product i is the primary product.
In contrast, alpha_0 is the ratio of customers landing on the webpage of a competitor.
We only consider the alpha ratios and disregard the total number of users. However, the alpha ratios will be subject to noise. 
That is, every day, the value of the alpha ratios will be realizations of independent Dirichlet random variables
"""
from typing import Dict, Tuple
from basic_types import CustomerTypeBased, CustomerClass
from Distribution import Dirichlet, PositiveIntegerGaussian, AbstractDistribution


class Environment:
    def __init__(self, alpha_distributions: CustomerTypeBased[Dirichlet]):
        self.day = 0
        self.alphas: Dict[CustomerClass, Tuple[float, ...]] = {k: () for k in CustomerClass}
        self.alpha_distributions: CustomerTypeBased[Dirichlet] = alpha_distributions
        self.new_day()

    def new_day(self):
        """
        Function that increments the day and updates the alpha ratios.
        TODO This function should also be able to udate the alpha ratios in the case of disagregation. Already included the aggregate_toggle in the environment class.
        """
        self.day += 1
        """
        Every day, the value of the alpha ratios of each product will be realizations of independent Dirichlet random variables.
        TODO Adjust the diriichlet distribution parameters. I have no idea what to put there
        """
        """
        It is mentioned online that dirichlet distributions are generalized form of alpha beta distribution.
        So, the 0th parameters is the amount of failure, and each alpha is the count of times we succeeded in selling.
        We haven't implemented this yet, but it is a dynamic distribution, we think.
        """
        self.alphas = {k: self.alpha_distributions[k].get_expectation() for k in CustomerClass}

    def get_current_day(self):
        return self.day

    def reset_day(self):
        self.day = 0

    def get_current_alpha(self, class_: CustomerClass):
        return self.alphas[class_]

    def get_expected_alpha(self, class_: CustomerClass):
        return self.alpha_distributions[class_].get_expectation()

    def get_aggregate_alpha(self, customer_counts: CustomerTypeBased[AbstractDistribution]):
        alphas = 0
        total = 0
        for customerClass in CustomerClass:
            count = customer_counts[customerClass].get_expectation()
            alphas = count * self.get_expected_alpha(customerClass)
            total += count
        return alphas / total
