import enum
from typing import List

from Distribution import Dirichlet, PositiveIntegerGaussian as PIG
from Product import Product

MAX_PRICE = 97 # There are missing values at index 99 and 100
LAMBDA_ = 0.1


products: List[Product] = [
    Product("T-shirt", [3, 12, 20, 40, 80]),
    Product("Shorts", [5, 13, 22, 35, 85]),
    Product("Towel", [2, 10, 15, 20, 35]),
    Product("Dumbbells", [5, 16, 34, 70, 100]),
    Product("Protein Powder", [15, 18, 25, 35, 60]),
]

# products[0].add_secondary_products(products[2], 0.4)
# products[1].add_secondary_products(products[0], 0.3, products[3], 0.1)
# products[2].add_secondary_products(products[1], 0.25, products[3], 0.05)
# products[3].add_secondary_products(products[0], 0.15)

from Environment import Environment
environment = Environment(Dirichlet([100, 100, 100, 100, 100, 100]))


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
