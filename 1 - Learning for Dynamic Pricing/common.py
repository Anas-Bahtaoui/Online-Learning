from typing import NamedTuple, Tuple, List

import numpy as np

lambda_ = 0.5
SECONDARY_OBSERVANCE_POSSIBILITIES = (1, lambda_)  # Second can change


class Customer(NamedTuple):
    reservation_price: List[Tuple[int, float]]
    class_: str


class Product(NamedTuple):
    id_: int
    name: str
    min_price: float
    max_price: float
    observance_probabilities: Tuple[int, float]


ProductTuple = Tuple[Product, Product, Product, Product, Product]
base_products: ProductTuple = (
    Product(id_=0, name="A", min_price=1.0, max_price=2.0, observance_probabilities=(0, 0)),
    Product(id_=1, name="B", min_price=1.0, max_price=2.0, observance_probabilities=(0, 0)),
    Product(id_=2, name="C", min_price=1.0, max_price=2.0, observance_probabilities=(0, 0)),
    Product(id_=3, name="D", min_price=1.0, max_price=2.0, observance_probabilities=(0, 0)),
    Product(id_=4, name="E", min_price=1.0, max_price=2.0, observance_probabilities=(0, 0)),
)


class Environment:
    day: int
    alpha: Tuple[float, float, float, float, float, float]
    _random_generator: np.random.Generator
    products: ProductTuple

    def __init__(self, fully_connected_graph: bool = False):
        self.day = 0
        self._random_generator = np.random.default_rng()
        products = list(base_products)
        for index, product in enumerate(products):
            distribution = iter(self._random_generator.dirichlet((len(base_products) - 1) * [1]))
            ## TODO: change, because probabilities don't need to add up to one
            # Also TODO: add the fully connected graph
            for other_product in products:
                if product.id_ == other_product.id_:
                    continue
                product = product._replace(observance_probabilities=(other_product.id_, next(distribution)))
            products[index] = product
        self.products = tuple(*products)
        self.new_day()

    def new_day(self):
        self.day += 1
        self.alpha = tuple(self._random_generator.dirichlet([1, 1, 1, 1, 1, 1]))


class PricingProduct(Product):
    prices = Tuple[float, float, float, float]


class Advertisement(NamedTuple):
    product: Product
    max_cost: float
