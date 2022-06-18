from typing import Tuple, Optional, List

import matplotlib.pyplot as plt
import networkx as nx

from entities import Product, Simulation, CustomerClass, SimulationConfig
from production import secondaries, product_configs, dirichlets

"""
Definition of the fully connected directed weighted graph. 
It is used to store the products as nodes and the click probabilities as edges. 
"""


class FullyConnectedGraph:
    def __init__(self, class_: CustomerClass, products: List[Product]):
        self.graph = nx.DiGraph()
        self.class_ = class_
        self._products = products
        self._add_products()
        self._add_edges()

    def _add_products(self):
        for product in self._products:
            self.graph.add_node(product, label=product.name)

    def _add_edges(self):
        """
        Each product can be connected to its two secondary products if is has secondary products.
        If the product has no secondary products, the function will not add any edges.
        """
        def _add_edge(tuple_: Optional[Tuple[Product, float]]):
            if tuple_ is not None:
                self.graph.add_edge(tuple_[0], self.class_, weight=tuple_[1])

        for product in self._products:
            _add_edge(product.secondary_products[self.class_][0])
            _add_edge(product.secondary_products[self.class_][1])

    def visualize(self):
        # TODO Adjust the names of the nodes and edges
        nx.draw(self.graph, with_labels=False)
        plt.show()


if __name__ == '__main__':
    # Create the graph
    simulation = Simulation(SimulationConfig(0, product_configs, secondaries, None, None, dirichlets), [])

    for class_ in CustomerClass:
        graph = FullyConnectedGraph(class_, simulation.products)
        graph.visualize()
