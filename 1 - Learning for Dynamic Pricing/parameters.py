import enum
from typing import List

from Distribution import Dirichlet, PositiveIntegerGaussian as PIG
from Product import ProductConfig, random_price_generator, Product

MAX_PRICE = 100
LAMBDA_ = 0.1

product_configs: List[ProductConfig] = [
    ProductConfig(name="Product 1", base_price=10, max_price=100),
    ProductConfig(name="Product 2", base_price=10, max_price=100),
    ProductConfig(name="Product 3", base_price=10, max_price=100),
    ProductConfig(name="Product 4", base_price=10, max_price=100),
    ProductConfig(name="Product 5", base_price=10, max_price=100),
]
# TODO: In the code fix the prices
# TODO: Make the product names and customer classes
# Take from https://www.notion.so/caspardietz/Project-Online-Learning-Applications-6880fbfb7d5445719c7ab0eed356f981?p=27728e0d751b4b4e9236d028c225a702
# Example 2


products: List[Product] = [Product(config, random_price_generator) for config in product_configs]

# products[0].add_secondary_products(products[2], 0.4)
# products[1].add_secondary_products(products[0], 0.3, products[3], 0.1)
# products[2].add_secondary_products(products[1], 0.25, products[3], 0.05)
# products[3].add_secondary_products(products[0], 0.15)

from Environment import Environment
environment = Environment(Dirichlet([100, 100, 100, 100, 100, 100]))


class CustomerClass(enum.IntEnum):
    A = 0
    B = 1
    C = 2


# TODO: is the Integer Gaussian good for this
# TODO: After we determine our products, update these values.

purchase_amounts = {
    CustomerClass.A: (PIG(5, 1), PIG(1, 1), PIG(3, 1), PIG(1, 1), PIG(1, 1)),
    CustomerClass.B: (PIG(8, 1), PIG(2, 1), PIG(6, 1), PIG(1, 1), PIG(2, 1)),
    CustomerClass.C: (PIG(15, 2), PIG(4, 1), PIG(8, 1), PIG(2, 1), PIG(8, 1)),
}

customer_counts = {
    CustomerClass.A: PIG(mean=50, variance=4),
    CustomerClass.B: PIG(mean=100, variance=6),
    CustomerClass.C: PIG(mean=30, variance=5),
}
