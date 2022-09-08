from entities import Dirichlet, CustomerTypeBased, PositiveIntegerGaussian as PIG, Constant
from production import dirichlets, customer_counts, purchase_amounts, secondaries

#dirichlet = Dirichlet([0.5, 0.5,  0.5, 0.5, 0.5, 0.5])
dirichlet = Dirichlet([1, 1,  1, 1, 1, 1])
same_diriclets: CustomerTypeBased[Dirichlet] = CustomerTypeBased(
    professional=dirichlet,
    young_beginner=dirichlet,
    old_beginner=dirichlet,
)

## TODO: Unnecessary, we have constant distribution :) (but hope nothing uses private values :()
small_variance = 0.0001
potential_diriclets = {
    "Production": dirichlets,
    "Same Weights": same_diriclets,
}
zeros = CustomerTypeBased(
    professional=Constant(0),
    young_beginner=Constant(0),
    old_beginner=Constant(0),
)
potential_customer_counts = {
    "Production": customer_counts,
    "Only Young": zeros._replace(young_beginner=customer_counts.young_beginner),
    "Only Old": zeros._replace(old_beginner=customer_counts.old_beginner),
    "Only Professional": zeros._replace(professional=customer_counts.professional),
    "Fixed": CustomerTypeBased(
        professional=PIG(customer_counts.professional.get_expectation(), small_variance),
        young_beginner=PIG(customer_counts.young_beginner.get_expectation(), small_variance),
        old_beginner=PIG(customer_counts.old_beginner.get_expectation(), small_variance),
    )
}

potential_purchase_amounts = {
    "Production": purchase_amounts,
    "All One": CustomerTypeBased(
        professional=[PIG(1, small_variance) for _ in range(5)],
        young_beginner=[PIG(1, small_variance) for _ in range(5)],
        old_beginner=[PIG(1, small_variance) for _ in range(5)],
    ),
    "Fixed": CustomerTypeBased(
        professional=[PIG(purchase_amounts.professional, small_variance) for _ in range(5)],
        young_beginner=[PIG(purchase_amounts.young_beginner, small_variance) for _ in range(5)],
        old_beginner=[PIG(purchase_amounts.old_beginner, small_variance) for _ in range(5)],
    ),
}
zero_secondaries = [[0 for _ in range(5)] for _ in range(5)]
potential_secondaries = {
    "Production": secondaries,
    "Disabled": CustomerTypeBased(
        professional=zero_secondaries,
        young_beginner=zero_secondaries,
        old_beginner=zero_secondaries,
    ),
}

potentials = {
    "Dirichlet value preset": potential_diriclets,
    "Customer count preset": potential_customer_counts,
    "Purchase amount preset": potential_purchase_amounts,
    "Secondary graph preset": potential_secondaries,
}
