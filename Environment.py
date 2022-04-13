'''
    Definition of the Environment class. It is used to iterate over the days of the simulation.
    
    Each day,there is a random number of potential customers. 
    Each product i is associated with a probability alpha_i, which is the ratio of customers landing on the webpage in which product i is the primary product.
    In contrast, alpha_0 is the ratio of customers landing on the webpage of a competitor.
    We only consider the alpha ratios and disregard the total number of users. However, the alpha ratios will be subject to noise. That is, every day, the value of the alpha ratios will be realizations of independent Dirichlet random variables
'''
#from distutils.command.config import config
from re import A
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

import Product
import Customer

lambda_ = 0.1 # The value of lambda is assumed to be known in all the three project proposals

class Environment:
    def __init__(self, agregate_toggle: bool = True):
        self.aggregate_toggle = agregate_toggle
        self.day = 0
        self.alpha = ()
        self.customers = pd.DataFrame(columns=['id', 'class', 'reservation_price'])
        self.products = pd.DataFrame(columns=['id', 'name', 'min_price', 'max_price', 'observance_probabilities'])
        self.products = self.products.append({'id': 0, 'name': 'A', 'min_price': 1.0, 'max_price': 2.0, 'observance_probabilities': (0, 0)}, ignore_index=True)
        self.products = self.products.append({'id': 1, 'name': 'B', 'min_price': 1.0, 'max_price': 2.0, 'observance_probabilities': (0, 0)}, ignore_index=True)
        self.products = self.products.append({'id': 2, 'name': 'C', 'min_price': 1.0, 'max_price': 2.0, 'observance_probabilities': (0, 0)}, ignore_index=True)
        self.products = self.products.append({'id': 3, 'name': 'D', 'min_price': 1.0, 'max_price': 2.0, 'observance_probabilities': (0, 0)}, ignore_index=True)
        self.products = self.products.append({'id': 4, 'name': 'E', 'min_price': 1.0, 'max_price': 2.0, 'observance_probabilities': (0, 0)}, ignore_index=True)
        self.new_day()
        
    def new_day(self):
        self.day += 1
        '''
            Every day, the value of the alpha ratios of each product will be realizations of independent Dirichlet random variables.
        '''
        self.alpha = tuple(np.random.dirichlet([1, 1, 1, 1, 1, 1], 1))
        
    def get_current_day(self):
        return self.day
    
    def reset_day(self):
        self.day = 0
    


'''
    Definition of the directed weighted graph. 
    It is used to store the products. 
    Each product has two child nodes, one for the primary product and one for the secondary product. 
    The weights are given, they are the click probabilities.
    There cannot be loops in the graph. 
'''
