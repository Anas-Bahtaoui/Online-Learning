from Product import ProductConfig, Product, random_price_generator
from Environment import Environment


def generate_sample_greedy_environment() -> Environment:
    # Create five products
    baseConfig = ProductConfig(id=-1, name="Base", base_price=1, max_price=10, production_cost=0.1)
    p1 = Product(baseConfig._replace(id=0, name="p1"), random_price_generator)
    p2 = Product(baseConfig._replace(id=1, name="p2"), random_price_generator)
    p3 = Product(baseConfig._replace(id=2, name="p3"), random_price_generator)
    p4 = Product(baseConfig._replace(id=3, name="p4"), random_price_generator)
    p5 = Product(baseConfig._replace(id=4, name="p5"), random_price_generator)

    # Assign secondary products to the products
    p1.add_secondary_products(p3, 0.4)
    p2.add_secondary_products(p1, 0.3, p4, 0.1)
    p3.add_secondary_products(p2, 0.25, p4, 0.05)
    p4.add_secondary_products(p1, 0.15)
    # Create the environment
    env = Environment()
    # Add the products to the environment
    env.add_product(p1)
    env.add_product(p2)
    env.add_product(p3)
    env.add_product(p4)
    env.add_product(p5)
    return env
