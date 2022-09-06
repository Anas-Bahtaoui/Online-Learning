import dataclasses
import enum
from typing import TypeVar, NamedTuple, List

import Distribution

MAX_PRICE = 97  # There are missing values at index 99 and 100
PRODUCT_COUNT = 5
PRICE_COUNT = 4


class Age(enum.Enum):
    YOUNG = "under 35"
    OLD = "over 35"


class Experience(enum.Enum):
    BEGINNER = "Beginner"
    PROFESSIONAL = "Professional"


class CustomerClass(enum.Enum):
    def __str__(self):
        return f"{self.value[0].value}{' ' + self.value[1].value if len(self.value) > 1 else ''}"

    PROFESSIONAL = (Experience.PROFESSIONAL,)
    YOUNG_BEGINNER = (Experience.BEGINNER, Age.YOUNG)
    OLD_BEGINNER = (Experience.BEGINNER, Age.OLD)


class ProductConfig(NamedTuple):
    name: str
    prices: List[float]


T = TypeVar("T")


class CustomerTypeBased(NamedTuple):
    professional: T
    young_beginner: T
    old_beginner: T

    def __getitem__(self, item: CustomerClass) -> T:
        if item == CustomerClass.PROFESSIONAL:
            return self.professional
        elif item == CustomerClass.YOUNG_BEGINNER:
            return self.young_beginner
        elif item == CustomerClass.OLD_BEGINNER:
            return self.old_beginner


@dataclasses.dataclass
class SimulationConfig:
    lambda_: float
    product_configs: List[ProductConfig]
    secondaries: CustomerTypeBased[List[List[float]]]
    purchase_amounts: CustomerTypeBased[List[Distribution.AbstractDistribution]]
    customer_counts: CustomerTypeBased[Distribution.AbstractDistribution]
    dirichlets: CustomerTypeBased[Distribution.Dirichlet]

    def __iter__(self):
        return iter(
            (self.lambda_, self.product_configs, self.secondaries, self.purchase_amounts, self.customer_counts, self
             .dirichlets))

    def __post_init__(self):

        # Check variables are correct

        # Products
        if len(self.product_configs) != PRODUCT_COUNT:
            raise Exception(f"The number of products is not {PRODUCT_COUNT}.")

        for product_config in self.product_configs:
            if len(product_config.prices) != PRICE_COUNT:
                raise Exception(f"The number of candidate prices of {product_config.name} is not {PRICE_COUNT}.")
            for price in product_config.prices:
                if price < 0 or price > MAX_PRICE:
                    raise Exception(
                        f"The price {price} of {product_config.name} is not in range [1, {MAX_PRICE}].")
        # Secondaries
        for class_ in CustomerClass:
            secondary_products = self.secondaries[class_]
            if len(secondary_products) != PRODUCT_COUNT or any(len(row) != PRODUCT_COUNT for row in secondary_products):
                raise Exception(
                    f"The secondary products need to match the number of products. ({PRODUCT_COUNT}x{PRODUCT_COUNT} in array size)")

            for from_ in secondary_products:
                for to_ in from_:
                    if to_ < 0 or to_ >= 1:
                        raise Exception(f"The probability of {to_} is not in range [0, 1).")
                # if sum(1 for to_ in from_ if to_ > 0) > 2:
                #     raise Exception(f"The number of non-zero probabilities of {from_} is not in range [0, 2].")
        # Lambda
        if self.lambda_ < 0 or self.lambda_ > 1:
            raise Exception(f"The lambda is not in range [0, 1].")
