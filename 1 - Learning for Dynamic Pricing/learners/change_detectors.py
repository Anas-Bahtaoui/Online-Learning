from typing import List

from entities import Customer


class ChangeDetectionAlgorithm:
    def has_changed(self, last_customers: List[Customer]) -> bool:
        # This approach works as only one detection for all cases
        # It is not arm specific
        raise NotImplementedError()

    def update(self, last_customers: List[Customer]):
        raise NotImplementedError()

    def reset(self):
        raise NotImplementedError()


class CumSum(ChangeDetectionAlgorithm):
    def __init__(self, M, eps, h):
        self.M = M
        self.eps = eps
        self.h = h
        self.reference = 0
        self.reset()

    # TODO: This implementation is bad, do instead: https://nbviewer.ipython.org/github/demotu/BMC/blob/master/notebooks/DetectCUSUM.ipynb

    def _calculate_sample(self, last_customers: List[Customer]) -> float:
        total_visits = 0
        total_purchases = 0
        for customer in last_customers:
            total_visits += len(customer.products_clicked)
            total_purchases += sum(1 if value > 0 else 0 for value in customer.products_bought.values())

        sample = total_purchases / total_visits

        return sample

    def update(self, last_customers: List[Customer]):
        sample = self._calculate_sample(last_customers)
        if self.t < self.M:
            self.reference += sample / self.M
        else:
            s_plus = (sample - self.reference) - self.eps
            s_minus = -(sample - self.reference) - self.eps
            self.g_plus = max(0, self.g_plus + s_plus)
            self.g_minus = max(0, self.g_minus + s_minus)


    def has_changed(self, last_customers: List[Customer]) -> bool:
        sample = self._calculate_sample(last_customers)
        self.t += 1
        if self.t <= self.M:
            return False
        else:
            s_plus = (sample - self.reference) - self.eps
            s_minus = -(sample - self.reference) - self.eps
            g_plus = max(0, self.g_plus + s_plus)
            g_minus = max(0, self.g_minus + s_minus)
            has_changed = g_plus > self.h or g_minus > self.h
            if has_changed:
                breakpoint()
            return has_changed

    def reset(self):
        self.t = 0
        # Because we call reset after one change has been detected.
        # So we don't want to lose the reference. but why? Isn't it better to have a new reference to start afresh?
        # TODO: original code doesn't have this line
        # self.reference = 0
        self.g_plus = 0
        self.g_minus = 0
