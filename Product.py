'''
    This is the definition of the Product class.
    Each product has four candidate prices that are between the base price and the maximum price. They candidate prices are equaly distributed.
'''
from numpy import product


class Product:
    def __init__(self, product_config):
        """
        @param product_config: product configuration dictionary.
        """
        self.name = product_config['name']
        self.base_price = product_config['base_price']
        self.max_price = product_config['max_price']
        self.candidate_prices = self.generate_candidate_prices()
        self.production_cost = product_config['production_cost']
        
        self.secondary_products = (product_config['secondary_products'][0], product_config['secondary_products'][1])
        
    def generate_candidate_prices(self):
        '''
            Generate four candidate prices that are eaully distributed between the max price and the base price.
        '''
        candidate_prices = []
        for i in range(4):
            candidate_prices.append(self.base_price + (self.max_price - self.base_price) * i / 3)
        return candidate_prices    


    def get_candidate_prices(self):
        '''
            @return: the candidate prices of the product.
        '''
        return self.candidate_prices
    
'''
    Function ProductFactory is used to create a new product.
'''
def ProductFactory(product_config):
    return Product(product_config)


'''
    Test the Product class
'''
myProduct = Product({'name': 'Product1', 'base_price': 1, 'max_price': 10, 'production_cost': 0.1, 'secondary_products': (0, 1)})