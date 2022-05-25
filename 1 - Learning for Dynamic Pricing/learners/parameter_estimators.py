from typing import List

from entities import Customer, CustomerTypeBased, AbstractDistribution, CustomerClass


class ParameterEstimator:
    def update(self, customer: Customer):
        raise NotImplementedError()

    def modify(self, criterias: List[float]) -> List[float]:
        raise NotImplementedError()


class AlphaEstimator(ParameterEstimator):
    def __init__(self):
        self.first_visit_counts = [0 for _ in range(5)]

    def update(self, customer: Customer):
        self.first_visit_counts[customer.products_clicked[0]] += 1

    def modify(self, criterias: List[float]) -> List[float]:
        return [criterias[i] * (self.first_visit_counts[i] / sum(self.first_visit_counts)) for i in range(5)]


class KnownAlphaEstimator(ParameterEstimator):
    def __init__(self, alpha: List[float]):
        self.alpha = alpha

    def update(self, customer: Customer):
        pass

    def modify(self, criterias: List[float]) -> List[float]:
        return [criterias[i] * self.alpha[i] for i in range(5)]


class NumberOfItemsSoldEstimator(ParameterEstimator):
    def __init__(self):
        self.product_buy_count = [0 for _ in range(5)]

    def update(self, customer: Customer):
        for product_i in customer.products_clicked:
            self.product_buy_count[product_i] += customer.products_bought[product_i]

    def modify(self, criterias: List[float]) -> List[float]:
        return [criterias[i] * (self.product_buy_count[i] / sum(self.product_buy_count)) for i in range(5)]


class KnownItemsSoldEstimator(ParameterEstimator):
    def __init__(self, customer_counts: CustomerTypeBased[AbstractDistribution],
                 purchase_amounts: CustomerTypeBased[List[AbstractDistribution]]):

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
        return [criterias[i] * self.n_items_sold[i] for i in range(5)]


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
        for i in range(5):
            normalized_secondary_visits[i][i] = 1.0
            for j in range(5):
                normalized_secondary_visits[i][j] = self._secondary_visit_counts[i][j] / self._total_visits[i]
        for to_ in range(5):
            total_prob = 0.0
            for from_ in range(5):
                total_prob += normalized_secondary_visits[from_][to_]
            criterias[to_] *= total_prob
        return criterias

    def __init__(self):
        # We don't need lambdas, they are already embedded in paths
        self._secondary_visit_counts = [[0 for _ in range(5)] for _ in range(5)]
        self._total_visits = [0 for _ in range(5)]


class KnownGraphWeightsEstimator(ParameterEstimator):
    def __init__(self, graph_weights: CustomerTypeBased[List[List[float]]]):
        self.graph_weights = graph_weights

    def update(self, customer: Customer):
        pass

    def modify(self, criterias: List[float]) -> List[float]:
        pass
