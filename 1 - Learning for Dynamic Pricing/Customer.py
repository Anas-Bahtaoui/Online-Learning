from collections import defaultdict
from functools import lru_cache
from typing import List, Dict, Set

import numpy as np
import scipy.stats

import Distribution
from Distribution import PositiveIntegerGaussian as PIG
from parameters import MAX_PRICE, CustomerClass

"""
    Each customer is belonging to one of the three customer classes. Each class an expected reservation price for each product.
    For each customers, the reservation price is a gaussian random variable with the mean and standard deviation of the expected reservation price.
"""


def reservation_price_distribution_from_curves(customer_class: CustomerClass, product_id: int, price: float) -> PIG:
    graph_result = read_conversion_probability(price, f"Demand Curves/curves/{customer_class.name}_{product_id}.npy")
    std_norm = scipy.stats.norm.ppf(1 - graph_result)
    sigma = 2  # TODO: Do we really want to always set the variance to two?
    # We at first wanted to fit this into 2 variances to cover 97 percent of the interval, but the sigma directly being the variance doesn't mean anything.
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
    if price > MAX_PRICE:
        price = MAX_PRICE
    # Sample from a linear function
    # 98.5 > look at 98 and 99 and interpolate the result in between
    return load_file(file_path)[round(price)][1]  # TODO: Linear interpolation maybe


"""
    This is the definition of the Customer class. There are three customers classes, distinguished by 2 binary features.
    Each customer belongs to a customer class.
    Each customer has a reservation price per product.
    For each customer we keep track of the products that they have clicked on.
    For each customer we keep track of the products that they have bought.
    The users classes potentially differ for the demand curves of the 5 products, number of daily users, alpha ratios, number of products sold, and graph probabilities
"""


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
