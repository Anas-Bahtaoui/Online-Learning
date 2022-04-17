import enum
import numpy as np

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


prices = {
    CustomerClass.A: (10.0, 300.0, 50.0, 1000.0, 100.0),
    CustomerClass.B: (12.0, 360.0, 60.0, 1200.0, 120.0),
    CustomerClass.C: (15.0, 450.0, 75.0, 1500.0, 150.0),
}


class Customer:
    def __init__(self, id_: int, class_: CustomerClass):
        """
        :param customer_config: customer configuration dictionary
        """
        self.id = id_
        self.class_ = class_
        self.products_clicked = []
        self.products_bought = []
        # The reservation price is a gaussian random variable with the mean and standard deviation of the expected reservation price of the customer class.
        self.reservation_prices = [price + np.random.normal(0, 1) for price in prices[self.class_]]

    def get_reservation_price_of(self, product_id: int) -> float:
        """
        Returns the reservation price of the product for the customer.
        
        :param product_id: product id
        :return: the reservation price of the product
        """
        return self.reservation_prices[product_id]

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
    customer = Customer(1, CustomerClass.A)
    print(customer.get_reservation_price_of(0))
    anotherCustomer = Customer(2, CustomerClass.B)
    print(anotherCustomer.get_reservation_price_of(0))
