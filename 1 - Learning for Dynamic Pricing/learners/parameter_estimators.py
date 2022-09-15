import copy
from operator import itemgetter
from typing import List, Tuple, TypeVar, Union, NamedTuple

from entities import Customer, CustomerTypeBased, AbstractDistribution, CustomerClass, Product, ObservationProbability

T = TypeVar("T")


class ParameterHistoryEntry(NamedTuple):
    incoming_prices: List[float]
    outgoing_prices: List[float]
    parameter: T


def safe_div(a, b):
    if b == 0:
        return 0
    return a / b


class ParameterEstimator:
    _history: List[ParameterHistoryEntry]

    def __init__(self):
        self._history = []

    def update(self, customer: Customer):
        raise NotImplementedError()

    def modify(self, criterias: List[float], *, register_history: bool = True) -> List[float]:
        raise NotImplementedError()

    def reset(self):
        pass


class AlphaEstimator(ParameterEstimator):
    def reset(self):
        self.first_visit_counts = [0 for _ in range(6)]

    def __init__(self):
        super().__init__()
        self.reset()

    def update(self, customer: Customer):
        if len(customer.products_clicked) > 0:
            self.first_visit_counts[customer.products_clicked[0]] += 1
        else:
            self.first_visit_counts[5] += 1

    def modify(self, criterias: List[float], register_history=True) -> List[float]:
        result = []
        sum_first_visit_counts = sum(self.first_visit_counts)
        ratios = []
        for i in range(5):
            first_visit_ratio = safe_div(self.first_visit_counts[i], sum_first_visit_counts)
            if first_visit_ratio == 0:
                breakpoint()
            result.append(criterias[i] * first_visit_ratio)
            ratios.append(first_visit_ratio)
        if register_history:
            self._history.append(ParameterHistoryEntry(criterias, result, ratios))
        return result


class KnownAlphaEstimator(ParameterEstimator):
    def __init__(self, alpha: List[float]):
        super().__init__()
        self.alpha = [item / sum(alpha[1:]) for item in alpha[1:]]  # Remove the probability of people leaving

    def update(self, customer: Customer):
        pass

    def modify(self, criterias: List[float], register_history=True) -> List[float]:
        result = [criterias[i] * self.alpha[i] for i in range(5)]
        if register_history:
            self._history.append(ParameterHistoryEntry(criterias, result, list(self.alpha)))
        return result


class NumberOfItemsSoldEstimator(ParameterEstimator):
    # This is actually correct, since it gets updated based on actual data

    def reset(self):
        self.product_buy_count = [0 for _ in range(5)]

    def __init__(self):
        super().__init__()
        self.reset()

    def update(self, customer: Customer):
        for product_i, count in customer.products_bought.items():
            self.product_buy_count[product_i] += count[0]

    def modify(self, criterias: List[float], register_history=True) -> List[float]:
        sum_product_buy_count = sum(self.product_buy_count)
        result = []
        ratios = []
        for i in range(5):
            product_ratio = safe_div(self.product_buy_count[i], sum_product_buy_count)
            # assert product_ratio != 0
            result.append(criterias[i] * product_ratio)
            ratios.append(product_ratio)
        if register_history:
            self._history.append(ParameterHistoryEntry(criterias, result, ratios))
        return result


class KnownItemsSoldEstimator(ParameterEstimator):
    def __init__(self, customer_counts: CustomerTypeBased[AbstractDistribution],
                 purchase_amounts: CustomerTypeBased[List[AbstractDistribution]]):
        # This is useless since because we can't observe classes and reservation prices
        # So this knowledge doesn't help
        super().__init__()
        total_customers = 0
        total_n_items_sold = [0 for _ in range(5)]
        for class_ in CustomerClass:
            count = customer_counts[class_].get_expectation()
            total_customers += count
            for product_id in range(5):
                total_n_items_sold[product_id] += count * purchase_amounts[class_][
                    product_id].get_expectation()
        sum_items_sold = sum(total_n_items_sold)
        total_prediction = [total_n_items_sold[i] / sum_items_sold for i in range(5)]
        self.items_sold_ratio = total_prediction

    def update(self, customer: Customer):
        pass

    def modify(self, criterias: List[float], register_history=True) -> List[float]:

        assert all(criteria != 0 for criteria in self.items_sold_ratio)
        result = [criterias[i] * self.items_sold_ratio[i] for i in range(5)]
        if register_history:
            self._history.append(ParameterHistoryEntry(criterias, result, list(self.items_sold_ratio)))
        return result


class GraphWeightsEstimator(ParameterEstimator):
    def update(self, customer: Customer):
        if len(customer.products_clicked) == 0:
            return
        self._total_visits[customer.products_clicked[0]] += 1
        if len(customer.products_clicked) < 2:
            return
        for i in range(len(customer.products_clicked) - 1):
            self._secondary_visit_counts[customer.products_clicked[i]][customer.products_clicked[i + 1]] += 1

    def modify(self, criterias: List[float], register_history=True) -> List[float]:
        normalized_secondary_visits = [[0.0 for _ in range(5)] for _ in range(5)]
        weights = [0.0 for _ in range(5)]
        for i in range(5):
            for j in range(5):
                normalized_secondary_visits[i][j] = safe_div(self._secondary_visit_counts[i][j], self._total_visits[i])
            normalized_secondary_visits[i][i] = 1.0
        for to_ in range(5):
            total_prob = 0.0
            for from_ in range(5):
                total_prob += normalized_secondary_visits[from_][to_]
            weights[to_] = total_prob

        result = [weights[i] * criterias[i] for i in range(5)]
        if register_history:
            self._history.append(ParameterHistoryEntry(criterias, result, weights))
        return criterias

    def reset(self):
        # We don't need lambdas, they are already embedded in paths
        self._secondary_visit_counts = [[0 for _ in range(5)] for _ in range(5)]
        self._total_visits = [0 for _ in range(5)]

    def __init__(self):
        super().__init__()
        self.reset()


class KnownGraphWeightsEstimator(ParameterEstimator):

    ### This doesn't work well because the theoretical weights are not the actual results
    # (the probability is problematic because after the first product, whether the subsequent products are
    # purchased also depends on reservation prices and classes)
    def __init__(self, graph_weights: CustomerTypeBased[List[List[float]]],
                 customer_counts: CustomerTypeBased[AbstractDistribution], lambda_: float):
        super().__init__()
        normalized_weights = [[0.0 for _ in range(5)] for _ in range(5)]
        young_c = customer_counts.young_beginner.get_expectation()
        old_c = customer_counts.old_beginner.get_expectation()
        prof_c = customer_counts.professional.get_expectation()
        total_customers = young_c + old_c + prof_c
        for i in range(5):
            for j in range(5):
                if i != j:
                    normalized_weights[i][j] = \
                        graph_weights.young_beginner[i][j] * young_c + \
                        graph_weights.old_beginner[i][j] * old_c + \
                        prof_c * graph_weights.professional[i][j]

                    normalized_weights[i][j] /= total_customers
        view_weight_matrix = [0.0 for _ in range(5)]

        def emulate_path(clicked_primaries: Tuple[int, ...], viewing_probability: float, current_id: int):
            if current_id in clicked_primaries:
                return
            view_weight_matrix[current_id] += viewing_probability
            first_p, second_p = sorted(enumerate(normalized_weights[current_id]), key=itemgetter(1), reverse=True)[:2]
            new_primaries = clicked_primaries + (current_id,)
            if first_p[1] > 0.0:
                emulate_path(new_primaries, first_p[1] * 1, first_p[0])
            if second_p[1] > 0.0:
                emulate_path(new_primaries, second_p[1] * lambda_, second_p[0])

        for product_id in range(5):
            emulate_path((), 1.0, product_id)
        view_weight_matrix = [item - 1 for item in view_weight_matrix]
        total = sum(view_weight_matrix)
        view_weight_matrix = [item / total for item in view_weight_matrix]
        view_weight_matrix = [item + 1 for item in view_weight_matrix]
        self.product_weights = view_weight_matrix

    def update(self, customer: Customer):
        pass

    def modify(self, criterias: List[float], register_history=True) -> List[float]:
        result = [criterias[i] * self.product_weights[i] for i in
                  range(5)]  # Since the weight indicates how much it will be visited in total, so multiply not divide
        if register_history:
            self._history.append(ParameterHistoryEntry(criterias, result, list(self.product_weights)))
        return result
