"""
    Definition of the Environment class. It is used to iterate over the days of the simulation.
    
    Each day,there is a random number of potential customers. 
    Each product i is associated with a probability alpha_i, which is the ratio of customers landing on the webpage in which product i is the primary product.
    In contrast, alpha_0 is the ratio of customers landing on the webpage of a competitor.
    We only consider the alpha ratios and disregard the total number of users. However, the alpha ratios will be subject to noise. That is, every day, the value of the alpha ratios will be realizations of independent Dirichlet random variables
"""
#from distutils.command.config import config
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

import Product
import Customer

class Environment:
    def __init__(self, agregate_toggle: bool = True):
        self.aggregate_toggle = agregate_toggle
        self.day = 0
        self.products = []
        self.alpha = ()
        self.customers = []
        self.new_day()
        self.lambda_ = 0.1 # The value of lambda is assumed to be known in all the three project proposals
    
    def add_product(self, product_config):
        """
        Function that adds products to the environment.
        """
        self.products.append(Product.ProductFactory(product_config))
       
    def add_product_to_env(self, product: Product):
        self.products.append(product) 
    
    def add_customer(self, customer_config):
        """
        Function that adds customers to the environment.
        """
        self.customers.append(Customer.CustomerFactory(customer_config))
    
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
        self.alpha = tuple(np.random.dirichlet([1, 1, 1, 1, 1, 1], 1))
        
    def get_current_day(self):
        return self.day
    
    def reset_day(self):
        self.day = 0
    
    def get_current_alpha(self):
        return self.alpha

"""
    Definition of the fully connected directed weighted graph. 
    It is used to store the products as nodes and the click probabilities as edges. 
"""
class FullyConnectedGraph:
    def __init__(self, environment: Environment):
        self.environment = environment
        self.graph = nx.DiGraph()
        self.add_products()
        #self.add_edges()
        
    def add_products(self):
        for product in self.environment.products:
            self.graph.add_node(product, label=product.name)
    
    def add_edges(self):
        """
        Each product can be connected to its two secondary products if is has secondary products.
        If the product has no secondary products, the function will not add any edges.
        """    
        for product in self.environment.products:
            if product.secondary_products[0] != None:
                self.graph.add_edge(product, product.secondary_products[0])
            if product.secondary_products[1] != None:
                self.graph.add_edge(product, product.secondary_products[1])
                
    def add_weight(self, primary_product: Product, secondary_product: Product, weight: float):
        """
        Function that adds a weight to a given edge between two products.
        :param primary_product: primary product.
        :param secondary_product: the secondary product.
        :param weight: the weight to be added.
        """
        self.graph.add_edge(primary_product.name, secondary_product.name, weight=weight)
   
    def get_current_graph(self):
        return self.graph
    
    def visualize(self):
        # TODO Adjust the names of the nodes and edges
        nx.draw(self.graph, with_labels=False)
        plt.show()
    

"""
    Test the Environment class.
""" 
if __name__ == '__main__':
    # Create five products
    p1 = Product.ProductFactory({'name': 'p1', 'base_price': 1, 'max_price': 10, 'production_cost': 0.1, 'secondary_products': (None, None)})
    p2 = Product.ProductFactory({'name': 'p2', 'base_price': 1, 'max_price': 10, 'production_cost': 0.1, 'secondary_products': (None, None)})
    p3 = Product.ProductFactory({'name': 'p3', 'base_price': 1, 'max_price': 10, 'production_cost': 0.1, 'secondary_products': (None, None)})
    p4 = Product.ProductFactory({'name': 'p4', 'base_price': 1, 'max_price': 10, 'production_cost': 0.1, 'secondary_products': (None, None)})
    p5 = Product.ProductFactory({'name': 'p5', 'base_price': 1, 'max_price': 10, 'production_cost': 0.1, 'secondary_products': (None, None)})
    # Assign secondary products to the products
    p1.add_secondary_products(p3, None)
    p2.add_secondary_products(p1, p4)
    p3.add_secondary_products(p2, p4)
    p4.add_secondary_products(p1, None)
    p5.add_secondary_products(None, None)
    # Create the environment
    env = Environment()
    # Add the products to the environment
    env.add_product_to_env(p1)
    env.add_product_to_env(p2)
    env.add_product_to_env(p3)
    env.add_product_to_env(p4)
    env.add_product_to_env(p5)
    # Create the graph
    graph = FullyConnectedGraph(env)
    graph.add_edges()
    graph.visualize()