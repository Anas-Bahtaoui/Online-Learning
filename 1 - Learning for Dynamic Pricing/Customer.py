import enum
import numpy as np
import scipy.stats

from Distribution import PositiveIntegerGaussian as PIG
from Product import Product

"""
    This is the definition of the Customer class. There are three customers classes, distinguished by 2 binary features.
    Each customer belongs to a customer class.
    Each customer has a reservation price per product.
    For each customer we keep track of the products that they have clicked on.
    For each customer we keep track of the products that they have bought.
    The users classes potentially differ for the demand curves of the 5 products, number of daily users, alpha ratios, number of products sold, and graph probabilities
"""

"""
    Each customer is belonging to one of the three customer classes. Each class an expected reservation price for each product.
    For each customers, the reservation price is a gaussian random variable with the mean and standard deviation of the expected reservation price.
"""


class CustomerClass(enum.IntEnum):
    A = 0
    B = 1
    C = 2


# These will come from the curve instead
# prices = {
#     CustomerClass.A: (10.0, 300.0, 50.0, 1000.0, 100.0),
#     CustomerClass.B: (12.0, 360.0, 60.0, 1200.0, 120.0),
#     CustomerClass.C: (15.0, 450.0, 75.0, 1500.0, 150.0),
# }

# TODO: is the integergaussian good for this
purchase_amounts = {
    CustomerClass.A: (PIG(5, 2), PIG(1, 1), PIG(3, 1), PIG(1, 1), PIG(1, 1)),
    CustomerClass.B: (PIG(8, 3), PIG(2, 1), PIG(6, 2), PIG(1, 1), PIG(2, 1)),
    CustomerClass.C: (PIG(15, 4), PIG(4, 1), PIG(8, 2), PIG(2, 1), PIG(8, 2)),
}

customer_counts = {
    CustomerClass.A: PIG(mean=50, variance=20),
    CustomerClass.B: PIG(mean=100, variance=20),
    CustomerClass.C: PIG(mean=10, variance=5),
}


def reservation_price_distribution_from_curves(customer_class: CustomerClass, product_id: int, price: float) -> PIG:
    graph_result = np.random.uniform(0, 1)  # Read from an actual curve
    std_norm = scipy.stats.norm.ppf(1 - graph_result)
    sigma = 2
    mu = price - sigma * std_norm
    return PIG(mu, sigma)

"""
Function that reads from the the demand curves (.npy files) and returns the conversion probability at a given price.
"""
def read_conversion_probability(price: float, file) -> float:
    return np.load(file)[price]

class Customer:
    def __init__(self, id_: int, class_: CustomerClass):
        """
        :param customer_config: customer configuration dictionary
        """
        self.id = id_
        self.class_ = class_
        self.products_clicked = []
        self.products_bought = []
        self.reservation_prices = [
            lambda price: reservation_price_distribution_from_curves(self.class_, product_id, price) for product_id in
            range(5)]
        self.purchase_amounts = [purchase_amounts[self.class_]]

    def get_reservation_price_of(self, product_id: int, product_price: float) -> PIG:
        """
        Returns the reservation price of the product for the customer.
        
        :param product_id: product id
        :param product_price: product price
        :return: the reservation price of the product
        """
        return self.reservation_prices[product_id](product_price)

    def get_class(self) -> CustomerClass:
        """
        :return: the customer class.
        """
        return self.class_

    def click_product(self, product_id):
        """
        Adds a product to the list of products that the customer has clicked on.
        
        :param product_id: product id.
        """
        self.products_clicked.append(product_id)

    def is_product_clicked(self, product_id):
        return product_id in self.products_clicked

    def buy_product(self, product_id):
        """
        :param product_id: product id.
        """
        self.products_bought.append(product_id)


"""
    Test the Customer class
"""
if __name__ == '__main__':
    # Create a new customers
    #customer = Customer(1, CustomerClass.A)
    #print(customer.get_reservation_price_of(0))
    #anotherCustomer = Customer(2, CustomerClass.B)
    #print(anotherCustomer.get_reservation_price_of(0))

    print(read_conversion_probability(20, 'test.npy'))