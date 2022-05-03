import enum
from collections import defaultdict
from functools import lru_cache
from typing import List, Tuple, Dict, Set

import numpy as np
import scipy.stats

import Distribution
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

# TODO: is the Integer Gaussian good for this
# TODO: After we determine our products, update these values.

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
    graph_result = read_conversion_probability(price, f"curves/{customer_class.name}_{product_id}.npy")
    std_norm = scipy.stats.norm.ppf(1 - graph_result)
    sigma = 2
    mu = price - sigma * std_norm
    return PIG(mu, sigma)


"""
Function that reads from the the demand curves (.npy files) and returns the conversion probability at a given price.
"""


@lru_cache(maxsize=None)
def load_file(file_path: str) -> np.array:
    return np.load(file_path)


def read_conversion_probability(price: float, file_path: str) -> float:
    # Prices we consider are in range 1-100
    if price > 100:
        price = 100
    # Sample from a linear function
    # 98.5 > look at 98 and 99 and interpolate the result in between
    return load_file(file_path)[round(price)][1]  # TODO: Linear interpolation maybe


class Customer:
    def __init__(self, class_: CustomerClass):
        """
        :param customer_config: customer configuration dictionary
        """
        self.class_ = class_
        self.products_clicked: Set[int] = set()
        self.products_bought: Dict[int, int] = defaultdict(int)
        self.reservation_prices: List[Distribution] = [
            lambda price: reservation_price_distribution_from_curves(self.class_, product_id, price) for product_id in
            range(5)]

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
        self.products_clicked.add(product_id)

    def is_product_clicked(self, product_id):
        return product_id in self.products_clicked

    def buy_product(self, product_id: int, product_count: int):
        """
        :param product_id: product id.
        """
        self.products_bought[product_id] += product_count


"""
    Test the Customer class
"""
if __name__ == '__main__':
    # Create a new customers
    # customer = Customer(1, CustomerClass.A)
    # print(customer.get_reservation_price_of(0))
    # anotherCustomer = Customer(2, CustomerClass.B)
    # print(anotherCustomer.get_reservation_price_of(0))

    print(read_conversion_probability(20, 'test.npy'))
