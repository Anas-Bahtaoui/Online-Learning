import enum
from typing import List, Tuple, Annotated, NamedTuple

from Distribution import Dirichlet, PositiveIntegerGaussian as PIG

MAX_PRICE = 97  # There are missing values at index 99 and 100
LAMBDA_ = 0.1


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


product_configs: List[ProductConfig] = [
    ProductConfig("T-shirt", [3, 12, 20, 40]),
    ProductConfig("Shorts", [5, 13, 22, 35]),
    ProductConfig("Towel", [2, 10, 15, 20]),
    ProductConfig("Dumbbells", [5, 16, 34, 70]),
    ProductConfig("Protein Powder", [15, 18, 25, 35]),

]
# @formatter:off
secondary_product_professional: List[List[float]] = [
    [0,    0,    0.4,  0,    0],
    [0.3,  0,    0,    0.1,  0],
    [0,    0.25, 0,    0.05, 0],
    [0.15, 0,    0,    0,    0],
    [0,    0,    0,    0,    0],
]
# TODO: After basic debugging is done, remove the following line.
secondary_product_professional: List[List[float]] = [[0]*5 for _ in range(5)]
# TODO: Custom values for these
secondary_product_beginner_young = secondary_product_professional
secondary_product_beginner_old = secondary_product_professional
# @formatter:on

secondaries = {
    CustomerClass.PROFESSIONAL: secondary_product_professional,
    CustomerClass.YOUNG_BEGINNER: secondary_product_beginner_young,
    CustomerClass.OLD_BEGINNER: secondary_product_beginner_old
}

# TODO: is the Integer Gaussian good for this
# TODO: After we determine our products, update these values.

purchase_amounts = {
    CustomerClass.PROFESSIONAL: (PIG(5, 1), PIG(1, 1), PIG(3, 1), PIG(1, 1), PIG(1, 1)),
    CustomerClass.YOUNG_BEGINNER: (PIG(8, 1), PIG(2, 1), PIG(6, 1), PIG(1, 1), PIG(2, 1)),
    CustomerClass.OLD_BEGINNER: (PIG(15, 2), PIG(4, 1), PIG(8, 1), PIG(2, 1), PIG(8, 1)),
}

customer_counts = {
    CustomerClass.PROFESSIONAL: PIG(mean=50, variance=4),
    CustomerClass.YOUNG_BEGINNER: PIG(mean=100, variance=6),
    CustomerClass.OLD_BEGINNER: PIG(mean=30, variance=5),
}
dirichlets = {
    CustomerClass.PROFESSIONAL: Dirichlet([100, 100, 100, 100, 100, 100]),
    CustomerClass.YOUNG_BEGINNER: Dirichlet([100, 100, 100, 100, 100, 100]),
    CustomerClass.OLD_BEGINNER: Dirichlet([100, 100, 100, 100, 100, 100]),
}
