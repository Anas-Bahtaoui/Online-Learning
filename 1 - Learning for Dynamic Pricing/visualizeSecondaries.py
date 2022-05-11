import matplotlib.pyplot as plt
import networkx as nx

from parameters import products

"""
Definition of the fully connected directed weighted graph. 
It is used to store the products as nodes and the click probabilities as edges. 
"""


class FullyConnectedGraph:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.add_products()
        self.add_edges()

    def add_products(self):
        for product in products:
            self.graph.add_node(product, label=product.name)

    def add_edges(self):
        """
        Each product can be connected to its two secondary products if is has secondary products.
        If the product has no secondary products, the function will not add any edges.
        """
        for product in products:
            if product.secondary_products[0] is not None:
                self.graph.add_edge(product.name, product.secondary_products[0][0].name, weight=product.secondary_products[0][1])
            if product.secondary_products[1] is not None:
                self.graph.add_edge(product.name, product.secondary_products[1][0].name, weight=product.secondary_products[1][1])

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
    # Create the graph
    graph = FullyConnectedGraph()
    graph.visualize()
