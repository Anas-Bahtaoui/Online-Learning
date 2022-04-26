"""
Definition of the Environment class. It is used to iterate over the days of the simulation.
    
Each day,there is a random number of potential customers. 
Each product i is associated with a probability alpha_i, which is the ratio of customers landing on the webpage in which product i is the primary product.
In contrast, alpha_0 is the ratio of customers landing on the webpage of a competitor.
We only consider the alpha ratios and disregard the total number of users. However, the alpha ratios will be subject to noise. 
That is, every day, the value of the alpha ratios will be realizations of independent Dirichlet random variables
"""
from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from Product import Product, ProductConfig, linear_price_generator
from Customer import Customer, CustomerClass

def n_users_of_class_generator(class_: CustomerClass) -> int:
    """
    Generates the number of users of a given class
    :param class_: the class of the users
    :return: the number of users of the given class
    """
    if class_ == CustomerClass.A:
        return np.random.poisson(5)
    elif class_ == CustomerClass.B:
        return np.random.poisson(10)
    elif class_ == CustomerClass.C:
        return np.random.poisson(20)


DIRICHLET_EXPECTATIONS = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]

def alpha_generator() -> Tuple[float, ...]:
    return tuple(np.random.dirichlet(np.array(DIRICHLET_EXPECTATIONS)))

def constant_generator() -> Tuple[float, ...]:
    return 1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6

class Environment:
    def __init__(self, alpha_generator=alpha_generator, aggregate_toggle: bool = True):
        self.aggregate_toggle = aggregate_toggle  # We first do the first 4, because this is tricky
        self.day = 0
        self.products: List[Product] = []
        self.alpha = ()
        self.alpha_generator = alpha_generator
        self.customers: List[Customer] = []
        self.new_day()
        self.lambda_ = 0.1  # The value of lambda is assumed to be known in all the three project proposals

    def add_product(self, product: Product):
        """
        Function that adds products to the environment.
        """
        self.products.append(product)

    def add_customer(self, customer: Customer):
        """
        Function that adds customers to the environment.
        """
        self.customers.append(customer)

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
        self.alpha = self.alpha_generator()

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

    def add_products(self):
        for product in self.environment.products:
            self.graph.add_node(product, label=product.name)

    def add_edges(self):
        """
        Each product can be connected to its two secondary products if is has secondary products.
        If the product has no secondary products, the function will not add any edges.
        """
        for product in self.environment.products:
            if product.secondary_products[0] is not None:
                self.graph.add_edge(product, product.secondary_products[0])
            if product.secondary_products[1] is not None:
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
    baseConfig = ProductConfig(id=1, name="Base", base_price=1, max_price=10, production_cost=0.1)
    p1 = Product(baseConfig._replace(name="p1"), linear_price_generator)
    p2 = Product(baseConfig._replace(name="p2"), linear_price_generator)
    p3 = Product(baseConfig._replace(name="p3"), linear_price_generator)
    p4 = Product(baseConfig._replace(name="p4"), linear_price_generator)
    p5 = Product(baseConfig._replace(name="p5"), linear_price_generator)

    # Assign secondary products to the products
    p1.add_secondary_products(p3, None)
    p2.add_secondary_products(p1, p4)
    p3.add_secondary_products(p2, p4)
    p4.add_secondary_products(p1, None)
    # Create the environment
    env = Environment()
    # Add the products to the environment
    env.add_product(p1)
    env.add_product(p2)
    env.add_product(p3)
    env.add_product(p4)
    env.add_product(p5)
    # Create the graph
    graph = FullyConnectedGraph(env)
    graph.add_edges()
    graph.visualize()
