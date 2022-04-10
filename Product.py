'''
Product class

@param name: name of the product
@param base_price: base price of the product
@param max_price: maximum price of the product
'''
class Product:
    def __init__(self, name, base_price, max_price):
        self.name = name
        self.base_price = base_price
        self.max_price = max_price

