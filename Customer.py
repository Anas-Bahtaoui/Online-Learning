import numpy as np
'''
    This is the definition of the Customer class. There are three customers classes, distinguished by 2 binaray features.
    Each customer belongs to a customer class.
    Each customer has a reservation price per product.
    For each customer we keep track of the products that they have clicked on.
    For each customer we keep track of the products that they have bought.
    The users classes potentially differ for the demand curves of the 5 products, number of daily users, alpha ratios, number of products sold, and graph probabilities
'''

'''
    Each customer is belonging to one of the three customer classes. Each class an expected reservation price for each product.
    For each customers, the reservation price is a gaussian random variable with the mean and standard deviation of the expected reservation price.
'''
ExpectedReservationPriceClassA = [10.0, 300.0, 50.0, 1000.0, 100.0]
ExpectedReservationPriceClassB = list(map(lambda x: x * 1.2, ExpectedReservationPriceClassA))
ExpectedReservationPriceClassC = list(map(lambda x: x * 1.5, ExpectedReservationPriceClassA))

class Customer:
    def __init__(self, customer_config):
        """
        @param customer_config: customer configuration dictionary.
        """
        self.id = customer_config['id']
        self.class_ = customer_config['class']
        # The reservation price is indidual each of the five products
        #self.reservation_prices = customer_config['reservation_prices']
        self.products_clicked = []
        self.products_bought = []
        if self.class_ == 'A':
            self.reservation_prices = [ExpectedReservationPriceClassA[i] + np.random.normal(0, 1) for i in range(5)]
        elif self.class_ == 'B':
            self.reservation_prices = [ExpectedReservationPriceClassB[i] + np.random.normal(0, 1) for i in range(5)]
        elif self.class_ == 'C':
            self.reservation_prices = [ExpectedReservationPriceClassC[i] + np.random.normal(0, 1) for i in range(5)]
        else:
            print('Error: UserClass is not A, B, or C')
            exit()
        self.products_clicked = []
        self.products_bought = []

    def get_reservation_price(self, product_id):
        """
        @param product_id: product id.
        @return: the reservation price of the product.
        """
        return self.reservation_prices[product_id]
    
    
    def get_class(self):
        """
        @return: the customer class.
        """
        return self.class_

    def update_products_clicked(self, product_id):
        """
        @param product_id: product id.
        """
        self.products_clicked.append(product_id)
    
    def buy_product(self, product_id):
        """
        @param product_id: product id.
        """
        self.products_bought.append(product_id)
    
'''
    Test the Customer class
'''
if __name__ == '__main__':
    #Create a new customers 
    customer = Customer({'id': 1, 'class': 'A'})
    print(customer.get_reservation_price(0))
    anotherCustomer = Customer({'id': 2, 'class': 'B'})
    print(anotherCustomer.get_reservation_price(0))