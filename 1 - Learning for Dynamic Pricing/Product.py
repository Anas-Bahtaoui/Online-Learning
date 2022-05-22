"""
    This is the definition of the Product class.
    Each product has four candidate prices that are between the base price and the maximum price. They candidate prices are equaly distributed.
"""
from operator import itemgetter
from typing import Tuple, List, Optional, Dict

from parameters import CustomerClass, product_configs, MAX_PRICE, secondaries

ObservationProbability = Tuple['Product', float]

last_product_id = -1


class Product:
    """
    Each product can have two secondary products. If the product has no secondary products, the value is None.
    Actually they can have more, but as stated in the text, only first two is significant.
    """
    secondary_products: Dict[CustomerClass, Tuple[Optional[ObservationProbability], Optional[ObservationProbability]]]

    def __init__(self, name: str, candidate_prices: List[float]):
        global last_product_id
        last_product_id += 1
        self.id = last_product_id
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


PRODUCT_COUNT = 5
PRICE_COUNT = 4

# Check variables are correct
if len(product_configs) != PRODUCT_COUNT:
    raise Exception(f"The number of products is not {PRODUCT_COUNT}.")

for product_config in product_configs:
    if len(product_config.prices) != PRICE_COUNT:
        raise Exception(f"The number of candidate prices of {product_config.name} is not {PRICE_COUNT}.")
    for price in product_config.prices:
        if price < 0 or price > MAX_PRICE:
            raise Exception(f"The price {price} of {product_config.name} is not in range [1, {MAX_PRICE}].")

for class_ in CustomerClass:
    secondary_products = secondaries[class_]
    if len(secondary_products) != PRODUCT_COUNT or any(len(row) != PRODUCT_COUNT for row in secondary_products):
        raise Exception(
            f"The secondary products need to match the number of products. ({PRODUCT_COUNT}x{PRODUCT_COUNT} in array size)")

    for from_ in secondary_products:
        for to_ in from_:
            if to_ < 0 or to_ >= 1:
                raise Exception(f"The probability of {to_} is not in range [0, 1).")
        if sum(1 for to_ in from_ if to_ > 0) > 2:
            raise Exception(f"The number of non-zero probabilities of {from_} is not in range [0, 2].")

# Create products

products = [Product(*product_config) for product_config in product_configs]
for class_ in CustomerClass:
    for from_, targets in enumerate(secondaries[class_]):
        first, second = sorted(enumerate(targets), key=itemgetter(1))[:2]
        if first[1] == 0:
            pass
        elif second[1] == 0:
            products[from_].add_secondary_products(class_, products[first[0]], first[1])
        else:
            products[from_].add_secondary_products(class_, products[first[0]], first[1], products[second[0]], second[1])

