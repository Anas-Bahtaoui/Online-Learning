import enum
from typing import List, Union, Dict, Tuple, NamedTuple, Optional

import numpy as np

from BanditLearner import BanditLearner
from basic_types import Experience, Age
from entities import Product
from itertools import product as prod


class BranchingLearner(BanditLearner):
    def __init__(self, config, learner_factory):
        super().__init__(config)
        self.config = config
        self._learner_factory = learner_factory
        self._reset_parameters()

    def _select_learner(self, e: Experience, a: Age):
        if (e, a) in self._learners:
            return self._learners[(e, a)]
        elif (e, None) in self._learners:
            return self._learners[(e, None)]
        elif (None, a) in self._learners:
            return self._learners[(None, a)]
        elif (None, None) in self._learners:
            return self._learners[(None, None)]
        else:
            raise RuntimeError("WTF?")

    def _select_price_criteria(self, product: Product) -> Dict[Tuple[Experience, Age], List[float]]:
        return {(e, a): self._select_learner(e, a)._select_price_criteria(product) for e, a in
                prod(list(Experience), list(Age))}

    def _update_learner_state(self, selected_price_indexes: Dict[Tuple[Experience, Age], List[int]], product_rewards,
                              t):

        customers = self._customer_history[-1]
        classes = {k: [] for k in selected_price_indexes.keys()}
        for customer in customers:
            classes[(customer.expertise, customer.age)].append(customer)

        for class_, customers_ in classes.items():
            self._select_learner(class_[0], class_[1])._update_learner_state(
                selected_price_indexes[(class_[0], class_[1])],
                self._calculate_product_rewards(selected_price_indexes, customers_),
                t)
        if t % 14 != 13:
            return

        bi_weekly_customer_history = self._customer_history[-14:]
        historical_customers = {k: [[] for _ in range(14)] for k in selected_price_indexes.keys()}
        for i, day in enumerate(bi_weekly_customer_history):
            for customer in day:
                historical_customers[(customer.expertise, customer.age)][i].append(customer)

        bi_weekly_experiment_history = self._experiment_history[-14:]

        def split_criteria(subset1, subset2) -> float:
            reward1 = self._calculate_product_rewards(selected_price_indexes, subset1)
            reward2 = self._calculate_product_rewards(selected_price_indexes, subset2)
            ratio1 = len(subset1) / len(customers)
            ratio1 -= 1.65 * np.sqrt(ratio1 * (1 - ratio1) / len(customers))
            ratio2 = len(subset2) / len(customers)
            ratio2 -= 1.65 * np.sqrt(ratio2 * (1 - ratio2) / len(customers))
            return sum(reward1) * ratio1 + sum(reward2) * ratio2

        def merge(a1, a2):
            return [a1[i] + a2[i] for i in range(14)]

        def split_learners(subset1, subset2):
            learner1 = self._learner_factory(self.config)
            learner2 = self._learner_factory(self.config)
            for i in range(14):
                price_indexes = bi_weekly_experiment_history[i][1]

                learner1._update_learner_state(price_indexes, self._calculate_product_rewards(price_indexes,
                                                                                              subset1[i]), i + 1)
                learner2._update_learner_state(price_indexes, self._calculate_product_rewards(price_indexes,
                                                                                              subset2[i]), i + 1)
            return learner1, learner2

        experience_split_beginner = classes[(Experience.BEGINNER, Age.YOUNG)] + classes[(Experience.BEGINNER, Age.OLD)]
        experience_split_pros = classes[(Experience.PROFESSIONAL, Age.YOUNG)] + classes[
            (Experience.PROFESSIONAL, Age.OLD)]

        split_experience_criteria = split_criteria(experience_split_beginner, experience_split_pros)
        age_split_young = classes[(Experience.BEGINNER, Age.YOUNG)] + classes[(Experience.PROFESSIONAL, Age.YOUNG)]
        age_split_old = classes[(Experience.BEGINNER, Age.OLD)] + classes[(Experience.PROFESSIONAL, Age.OLD)]

        split_age_criteria = split_criteria(age_split_young, age_split_old)
        base_criteria = sum(product_rewards)
        if split_experience_criteria > base_criteria and (
                split_age_criteria <= base_criteria or split_experience_criteria > split_age_criteria):
            beginner_learner, professional_learner = split_learners(merge(historical_customers[(Experience.BEGINNER,Age.YOUNG)], historical_customers[(Experience.BEGINNER,Age.OLD)]), merge(historical_customers[(Experience.PROFESSIONAL,Age.YOUNG)], historical_customers[(Experience.PROFESSIONAL,Age.OLD)]))
            split_age_beginner = split_criteria(classes[(Experience.BEGINNER, Age.YOUNG)],
                                                classes[(Experience.BEGINNER, Age.OLD)]) > sum(
                self._calculate_product_rewards(selected_price_indexes, experience_split_beginner))
            split_age_pros = split_criteria(classes[(Experience.PROFESSIONAL, Age.YOUNG)],
                                            classes[(Experience.PROFESSIONAL, Age.OLD)]) > base_criteria > sum(
                self._calculate_product_rewards(selected_price_indexes, experience_split_pros))
            if split_age_beginner:
                self._learners[(Experience.BEGINNER, Age.YOUNG)], self._learners[
                    (Experience.BEGINNER, Age.OLD)] = split_learners(
                    historical_customers[(Experience.BEGINNER, Age.YOUNG)], historical_customers[(Experience.BEGINNER, Age.OLD)])
            else:
                self._learners[(Experience.BEGINNER, None)] = beginner_learner
            if split_age_pros:
                self._learners[(Experience.PROFESSIONAL, Age.YOUNG)], self._learners[
                    (Experience.PROFESSIONAL, Age.OLD)] = split_learners(
                    historical_customers[(Experience.PROFESSIONAL, Age.YOUNG)], historical_customers[(Experience.PROFESSIONAL, Age.OLD)])
            else:
                self._learners[(Experience.PROFESSIONAL, None)] = professional_learner
        elif split_age_criteria > base_criteria:
            young_learner, old_learner = split_learners(merge(historical_customers[(Experience.BEGINNER,Age.YOUNG)], historical_customers[(Experience.PROFESSIONAL,Age.YOUNG)]), merge(historical_customers[(Experience.BEGINNER,Age.OLD)], historical_customers[(Experience.PROFESSIONAL,Age.OLD)]))
            split_experience_young = split_criteria(classes[(Experience.BEGINNER, Age.YOUNG)],
                                                    classes[(Experience.PROFESSIONAL, Age.YOUNG)]) > sum(
                self._calculate_product_rewards(selected_price_indexes, age_split_young))
            split_experience_old = split_criteria(classes[(Experience.BEGINNER, Age.OLD)],
                                                  classes[(Experience.PROFESSIONAL, Age.OLD)]) > base_criteria > sum(
                self._calculate_product_rewards(selected_price_indexes, age_split_old))
            if split_experience_young:
                self._learners[(Experience.BEGINNER, Age.YOUNG)], self._learners[
                    (Experience.PROFESSIONAL, Age.YOUNG)] = split_learners(
                    historical_customers[(Experience.BEGINNER, Age.YOUNG)], historical_customers[(Experience.PROFESSIONAL, Age.YOUNG)])
            else:
                self._learners[(None, Age.YOUNG)] = young_learner
            if split_experience_old:
                self._learners[(Experience.BEGINNER, Age.OLD)], self._learners[
                    (Experience.PROFESSIONAL, Age.OLD)] = split_learners(
                    historical_customers[(Experience.BEGINNER, Age.OLD)], historical_customers[(Experience.PROFESSIONAL, Age.OLD)])
            else:
                self._learners[(None, Age.OLD)] = old_learner

    def _reset_parameters(self):
        new_learner = self._learner_factory(self.config)
        if hasattr(self, "_vars"):
            new_learner.refresh_vars(*self._vars)
            new_learner._reset_parameters()
        self._learners: Dict[Tuple[Optional[Experience], Optional[Age]], BanditLearner] = {
            (None, None): new_learner}

    def refresh_vars(self, _products, _environment, _config):
        self._vars = (_products, _environment, _config)
        self._environment = _environment
        self._config = _config
        self._products = _products
        [learner.refresh_vars(*self._vars) for learner in self._learners.values()]

    def reset(self):
        self.__init__(self.config, self._learner_factory)