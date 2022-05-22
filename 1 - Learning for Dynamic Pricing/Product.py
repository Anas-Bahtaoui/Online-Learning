"""
    This is the definition of the Product class.
    Each product has four candidate prices that are between the base price and the maximum price. They candidate prices are equaly distributed.
"""
from typing import Tuple, List, Optional

ObservationProbability = Tuple['Product', float]

last_product_id = -1


class Product:
    """
    Each product can have two secondary products. If the product has no secondary products, the value is None.
    Actually they can have more, but as stated in the text, only first two is significant.
    """
    secondary_products: Tuple[Optional[ObservationProbability], Optional[ObservationProbability]]

    def __init__(self, name: str, candidate_prices: List[float]):
        global last_product_id
        last_product_id += 1
        self.id = last_product_id
        self.name = name
        self.candidate_prices: List[float] = candidate_prices
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
