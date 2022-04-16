import numpy as np
from scipy.stats import bernoulli
"""
    This is the definition of the Customer class. There are three customers classes, distinguished by 2 binaray features.
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
ExpectedReservationPriceClassA = [10.0, 300.0, 50.0, 1000.0, 100.0]
ExpectedReservationPriceClassB = list(map(lambda x: x * 1.2, ExpectedReservationPriceClassA)) # "...Class B has 20% more money to spend on products compared to class A"
ExpectedReservationPriceClassC = list(map(lambda x: x * 1.5, ExpectedReservationPriceClassA))

class Customer:
    def __init__(self, customer_config):
        """
        :param customer_config: customer configuration dictionary
        """
        self.id = customer_config['id']
        self.class_ = customer_config['class']
        self.products_clicked = []
        self.products_bought = []
        # The reservation price is a gaussian random variable with the mean and standard deviation of the expected reservation price of the customer class.
        if self.class_ == 'A':
            self.reservation_prices = [ExpectedReservationPriceClassA[i] + np.random.normal(0, 1) for i in range(5)]
        elif self.class_ == 'B':
            self.reservation_prices = [ExpectedReservationPriceClassB[i] + np.random.normal(0, 1) for i in range(5)]
        elif self.class_ == 'C':
            self.reservation_prices = [ExpectedReservationPriceClassC[i] + np.random.normal(0, 1) for i in range(5)]
        else:
            print('Error: UserClass is not A, B, or C')
            exit()
        #self.products_clicked = [] # Used to make sure that we don't show the same product twice to the same customer.
        #self.products_bought = []

    def get_reservation_price(self, product_id):
        """
        Returns the reservation price of the product for the customer.
        
        :param product_id: product id
        :return: the reservation price of the product
        """
        return self.reservation_prices[product_id]
    
    
    def get_class(self):
        """
        :return: the customer class.
        """
        return self.class_

    def update_products_clicked(self, product_id):
        """
        Adds a product to the list of products that the customer has clicked on.
        
        :param product_id: product id.
        """
        self.products_clicked.append(product_id)
    
    def clicked_on_product(self, product_id):
        """
        Returns True if the customer has clicked on the product.
        
        :param product_id: product id.
        :return: True if the product has been clicked on by the customer.
        """
        return product_id in self.products_clicked
    
    def buy_product(self, product_id):
        """
        :param product_id: product id.
        """
        self.products_bought.append(product_id)
  
"""
    CustomerFactory is used to create a new customer.
"""
def CustomerFactory(customer_config):
    return Customer(customer_config)


"""
    Test the Customer class
"""
if __name__ == '__main__':
    #Create a new customers 
    customer = Customer({'id': 1, 'class': 'A'})
    print(customer.get_reservation_price(0))
    anotherCustomer = Customer({'id': 2, 'class': 'B'})
    print(anotherCustomer.get_reservation_price(0))