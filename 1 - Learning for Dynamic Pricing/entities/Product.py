"""
    This is the definition of the Product class.
    Each product has four candidate prices that are between the base price and the maximum price. They candidate prices are equaly distributed.
"""
from typing import Tuple, List, Optional, Dict

from basic_types import CustomerClass

ObservationProbability = Tuple['Product', float]


class Product:
    """
    Each product can have two secondary products. If the product has no secondary products, the value is None.
    Actually they can have more, but as stated in the text, only first two is significant.
    """
    secondary_products: Dict[CustomerClass, Tuple[Optional[ObservationProbability], Optional[ObservationProbability]]]

    def __init__(self, product_id: int, name: str, candidate_prices: List[float]):
        self.id = product_id
        self.name = name
        self.candidate_prices: List[float] = candidate_prices
        self.secondary_products = {k: (None, None) for k in CustomerClass}

    def get_candidate_prices(self):
        """
        :return: the candidate prices of the product.
        """
        return self.candidate_prices

    def add_secondary_products(self, class_: CustomerClass, secondary_product_1: 'Product', prob1: float,
                               secondary_product_2: Optional['Product'] = None, prob2: float = 0):
        """
        Add secondary products to the product. If the product has no secondary products, the function stores None.
        """
        self.secondary_products[class_] = (
            (secondary_product_1, prob1), (secondary_product_2, prob2) if secondary_product_2 is not None else None)

    def serialize(self):
        return {
            "id": self.id,
            "name": self.name,
            "candidate_prices": self.candidate_prices,
            # We don't serialize secondary products, maybe in the future
        }
