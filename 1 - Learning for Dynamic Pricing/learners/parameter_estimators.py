import copy
from operator import itemgetter
from typing import List, Tuple, TypeVar, Union, NamedTuple

from entities import Customer, CustomerTypeBased, AbstractDistribution, CustomerClass, Product, ObservationProbability

T = TypeVar("T")


class HistoryEntry(NamedTuple):
    incoming_prices: List[float]
    outgoing_prices: List[float]
    parameter: T


def safe_div(a, b):
    if b == 0:
        return 0
    return a / b


class ParameterEstimator:
    _history: List[HistoryEntry]
    def __init__(self):
        self._history = []

    def update(self, customer: Customer):
        raise NotImplementedError()

    def modify(self, criterias: List[float]) -> List[float]:
        raise NotImplementedError()


class AlphaEstimator(ParameterEstimator):
    def __init__(self):
        super().__init__()
        self.first_visit_counts = [0 for _ in range(5)]

    def update(self, customer: Customer):
        if len(customer.products_clicked) > 0:
            self.first_visit_counts[customer.products_clicked[0]] += 1

    def modify(self, criterias: List[float]) -> List[float]:
        result = [safe_div(criterias[i], 1 + safe_div(self.first_visit_counts[i], sum(self.first_visit_counts))) for i in
                  range(5)]
        self._history.append(HistoryEntry(criterias, result, list(self.first_visit_counts)))
        return result


class KnownAlphaEstimator(ParameterEstimator):
    def __init__(self, alpha: List[float]):
        super().__init__()
        self.alpha = alpha

    def update(self, customer: Customer):
        pass

    def modify(self, criterias: List[float]) -> List[float]:
        result = [ safe_div(criterias[i], self.alpha[i]) for i in range(5)]
        self._history.append(HistoryEntry(criterias, result, list(self.alpha)))
        return result


class NumberOfItemsSoldEstimator(ParameterEstimator):
    def __init__(self):
        super().__init__()
        self.product_buy_count = [0 for _ in range(5)]

    def update(self, customer: Customer):
        for product_i in customer.products_clicked:
            self.product_buy_count[product_i] += customer.products_bought[product_i]

    def modify(self, criterias: List[float]) -> List[float]:
        result = [safe_div(criterias[i], 1 + safe_div(self.product_buy_count[i], sum(self.product_buy_count))) for i in
                  range(5)]
        self._history.append(HistoryEntry(criterias, result, list(self.product_buy_count)))
        return result


class KnownItemsSoldEstimator(ParameterEstimator):
    def __init__(self, customer_counts: CustomerTypeBased[AbstractDistribution],
                 purchase_amounts: CustomerTypeBased[List[AbstractDistribution]]):

        super().__init__()
        total_customers = 0
        total_n_items_sold = [0 for _ in range(5)]
        for class_ in CustomerClass:
            count = customer_counts[class_].get_expectation()
            total_customers += count
            for product_id in range(5):
                total_n_items_sold[product_id] += count * purchase_amounts[class_][
                    product_id].get_expectation()
        total_prediction = [total_n_items_sold[i] / total_customers for i in range(5)]
        self.n_items_sold = total_prediction

    def update(self, customer: Customer):
        pass

    def modify(self, criterias: List[float]) -> List[float]:
        result = [safe_div(criterias[i], self.n_items_sold[i]) for i in range(5)]
        self._history.append(HistoryEntry(criterias, result, list(self.n_items_sold)))
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

    def modify(self, criterias: List[float]) -> List[float]:
        normalized_secondary_visits = [[0.0 for _ in range(5)] for _ in range(5)]
        old_criterias = copy.copy(criterias)
        for i in range(5):
            normalized_secondary_visits[i][i] = 1.0
            for j in range(5):
                normalized_secondary_visits[i][j] = safe_div(self._secondary_visit_counts[i][j], self._total_visits[i])
        for to_ in range(5):
            total_prob = 0.0
            for from_ in range(5):
                total_prob += normalized_secondary_visits[from_][to_]
            criterias[to_] = safe_div(criterias[to_], 1 + total_prob)
        self._history.append(
            HistoryEntry(old_criterias, criterias, [list(elem) for elem in self._secondary_visit_counts]))
        return criterias

    def __init__(self):
        # We don't need lambdas, they are already embedded in paths
        super().__init__()
        self._secondary_visit_counts = [[0 for _ in range(5)] for _ in range(5)]
        self._total_visits = [0 for _ in range(5)]


class KnownGraphWeightsEstimator(ParameterEstimator):
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

        def emulate_path(clicked_primaries: Tuple[int, ...], viewing_probability: float, current_id: int):
            if current_id in clicked_primaries:
                return 0
            result_ = viewing_probability
            first_p, second_p = sorted(enumerate(normalized_weights[current_id]), key=itemgetter(1))[:2]
            new_primaries = clicked_primaries + (current_id,)
            if first_p[1] > 0.0:
                result_ += emulate_path(new_primaries, first_p[1] * 1, first_p[0])
            if second_p[1] > 0.0:
                result_ += emulate_path(new_primaries, second_p[1] * lambda_, second_p[0])
            return result_

        self.product_weights = [emulate_path((), 1.0, i) for i in range(5)]

    def update(self, customer: Customer):
        pass

    def modify(self, criterias: List[float]) -> List[float]:
        result = [safe_div(criterias[i], self.product_weights[i]) for i in range(5)]
        self._history.append(HistoryEntry(criterias, result, list(self.product_weights)))
        return result
