from typing import List, NamedTuple

from entities import Customer


class ChangeHistoryItem(NamedTuple):
    g_plus: float
    g_minus: float
    sample: float
    threshold: float


class ChangeDetectionAlgorithm:
    def has_changed(self) -> bool:
        # This approach works as only one detection for all cases
        # It is not arm specific
        raise NotImplementedError()

    def update(self, last_customers: List["Customer"]) -> NamedTuple:
        raise NotImplementedError()

    def update_experiment_days(self, n_days: int) -> None:
        raise NotImplementedError()

    def reset(self):
        raise NotImplementedError()


from math import comb, log, ceil, floor


class CumSum(ChangeDetectionAlgorithm):
    def __init__(self, M, e, n_breakpoints):
        self.M = M
        self.e = e
        c1_plus = log( (4 * e / ((1 - e) ** 2)) * comb(M, ceil(2 * e * M)) * (2 * e) ** M + 1)
        c1_minus = log( (4 * e / ((1 - e) ** 2)) * comb(M, floor(2 * e * M)) * (2 * e) ** M + 1)
        self.c1 = min(c1_plus, c1_minus)
        self.n_breakpoints = n_breakpoints
        self.reset()

    def update_experiment_days(self, n_days: int) -> None:
        self.threshold = 0.3 # 1 / self.c1 * log(n_days / self.n_breakpoints)

    def _calculate_sample(self, last_customers: List["Customer"]) -> float:
        total_visits = 0
        total_purchases = 0
        for customer in last_customers:
            total_visits += len(customer.products_clicked)
            total_purchases += sum(1 if value[0] > 0 else 0 for value in customer.products_bought.values())

        sample = total_purchases / total_visits

        return sample

    def update(self, last_customers: List["Customer"]) -> ChangeHistoryItem:

        sample = self._calculate_sample(last_customers)
        history_item = ChangeHistoryItem(0, 0, sample, self.threshold)
        if self.t < self.M:
            self.samples.append(sample)
            self.u_0 += sample / self.M
            self.alerts.append(False)
        else:
            s_plus = sample - self.u_0 - self.e
            s_minus = self.u_0 - sample - self.e
            g_plus = max(self.g_pluses[-1] + s_plus, 0)
            g_minus = max(self.g_minuses[-1] + s_minus, 0)
            alert = g_plus > self.threshold or g_minus > self.threshold
            if alert:
                g_plus = 0
                g_minus = 0
            self.samples.append(sample)
            self.g_pluses.append(g_plus)
            self.g_minuses.append(g_minus)
            self.alerts.append(alert)
            history_item = ChangeHistoryItem(g_plus, g_minus, sample, self.threshold)
        self.t += 1
        return history_item

    def has_changed(self) -> bool:
        has_changed = self.alerts[-1]
        return has_changed

    def reset(self):
        self.t = 0
        self.g_pluses = [0.0] * self.M
        self.g_minuses = [0.0] * self.M
        self.alerts = []
        self.samples = []
        self.u_0 = 0
