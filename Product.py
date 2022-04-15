'''
    This is the definition of the Product class.
    Each product has four candidate prices that are between the base price and the maximum price. They candidate prices are equaly distributed.
'''
from numpy import linspace, product


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
        # Each product can have two secondary products. If the product has no secondary products, the value is None.
        self.secondary_products = product_config['secondary_products']
    
    def generate_candidate_prices(self):
        '''
            Generate four candidate prices that are eaully distributed between the max price and the base price.
        '''
        candidate_prices = []
        candidate_prices = linspace(self.base_price, self.max_price, 4)
        return candidate_prices    

    def get_candidate_prices(self):
        '''
            @return: the candidate prices of the product.
        '''
        return self.candidate_prices
    
    '''
        Add secondary products to the product. If the product has no secondary products, the function stores None.
    '''
    def add_secondary_products(self, secondary_product_1, secondary_product_2):
        self.secondary_products = (secondary_product_1, secondary_product_2)
        
'''
    Function ProductFactory is used to create a new product.
'''
def ProductFactory(product_config):
    return Product(product_config)

'''
    Test the Product class
'''
# Product with no secondary products.
prod = ProductFactory({'name': 'Product1', 'base_price': 1, 'max_price': 10, 'production_cost': 0.1, 'secondary_products': (None, None)})
# Product with secondary products.
prod2 = ProductFactory({'name': 'Product1', 'base_price': 1, 'max_price': 10, 'production_cost': 0.1, 'secondary_products': (prod, None)})