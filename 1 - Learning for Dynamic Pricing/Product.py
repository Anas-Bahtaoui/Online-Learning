"""
    This is the definition of the Product class.
    Each product has four candidate prices that are between the base price and the maximum price. They candidate prices are equaly distributed.
"""
from typing import Tuple, List, NamedTuple, Optional, Callable

from numpy import linspace
from Distribution import PositiveIntegerGaussian as PIG


class ProductConfig(NamedTuple):
    """
    This is the definition of the ProductConfig class.
    It contains the product name, the base price, the maximum price, and the number of candidate prices.
    """
    name: str
    base_price: float
    max_price: float
    production_cost: float


PriceGenerator = Callable[[float, float], List[float]]

ObservationProbability = Tuple['Product', float]

last_product_id = -1


class Product:
    """
    Each product can have two secondary products. If the product has no secondary products, the value is None.
    Actually they can have more, but as stated in the text, only first two is significant.
    """
    secondary_products: Tuple[Optional[ObservationProbability], Optional[ObservationProbability]]

    def __init__(self, product_config: ProductConfig, generator: PriceGenerator):
        """
        :param product_config: product configuration dictionary.
        """
        global last_product_id
        last_product_id += 1
        self.id = last_product_id
        self.name, self.base_price, self.max_price, self.production_cost = product_config
        self.candidate_prices: List[float] = sorted(generator(self.base_price, self.max_price))
        self.secondary_products = (None, None)

    def get_candidate_prices(self):
        """
        :return: the candidate prices of the product.
        """
        return self.candidate_prices

    def add_secondary_products(self, secondary_product_1: 'Product', prob1: float,
                               secondary_product_2: Optional['Product'] = None, prob2: float = 0):
        """
        Add secondary products to the product. If the product has no secondary products, the function stores None.
        """
        self.secondary_products = (
            (secondary_product_1, prob1), (secondary_product_2, prob2) if secondary_product_2 is not None else None)


def linear_price_generator(base_price: float, max_price: float) -> List[float]:
    """
    :param base_price: the base price of the product.
    :param max_price: the maximum price of the product.
    :return: a list of four candidate prices that are equally distributed between the max price and the base price.
    """
    return list(linspace(base_price, max_price, 4))


def random_price_generator(base_price: float, max_price: float) -> List[float]:
    distr = PIG(base_price, (max_price - base_price) / 3)
    # We want 99.7 of the prices to be between the base price and the max price.

    def sample() -> float:
        samp = max_price
        while samp >= max_price or samp <= base_price:
            samp = distr.get_sample_value()
        return samp

    return sorted(sample() for _ in range(4))
