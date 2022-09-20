import enum
from typing import List, Union, Dict, Tuple, NamedTuple, Optional

import numpy as np

from BanditLearner import BanditLearner
from basic_types import Experience, Age
from entities import Product
from itertools import product as prod


class BranchingLearner(BanditLearner):
    def _upper_bound(self):
        return 0

    def __init__(self, config, learner_factory):
        super().__init__(config)
        self.config = config
        self._learner_factory = learner_factory
        self._reset_parameters()

    def _reset_parameters(self):
        new_learner = self._learner_factory(self.config)
        if hasattr(self, "_vars"):
            new_learner.refresh_vars(*self._vars)
            new_learner._reset_parameters()
        self._learners: Dict[Tuple[Optional[Experience], Optional[Age]], BanditLearner] = {
            (None, None): new_learner}

    def refresh_vars(self, _products, _environment, _config):
        super(BranchingLearner, self).refresh_vars(_products, _environment, _config)
        self._vars = (_products, _environment, _config)
        self._environment = _environment
        self._config = _config
        self._products = _products
        [learner.refresh_vars(*self._vars) for learner in self._learners.values()]

    def reset(self):
        self.__init__(self.config, self._learner_factory)

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
        x = {}
        ran_for = set()

        for exp in list(Experience):
            for age in list(Age):
                if (exp, age) in self._learners:
                    x[(exp, age)] = self._select_learner(exp, age)._select_price_criteria(product)
                    ran_for.add((exp, age))
                elif (exp, None) in self._learners and exp not in ran_for:
                    x[(exp, age)] = self._select_learner(exp, age)._select_price_criteria(product)
                    x[(exp, [age for age in list(Age) if age != age][0])] = x[(exp, age)]
                    ran_for.add(exp)
                elif (None, age) in self._learners and age not in ran_for:
                    x[(exp, age)] = self._select_learner(exp, age)._select_price_criteria(product)
                    x[([exp for exp in list(Experience) if exp != exp][0], age)] = x[(exp, age)]
                    ran_for.add(age)

        if not ran_for:
            # There is only the default learner, no splitting yet
            val = self._learners[(None, None)]._select_price_criteria(product)
            x = {(e, a): val for e, a in prod(list(Experience), list(Age))}
        return x

    def _update_learner_state(self, selected_price_indexes: Dict[Tuple[Experience, Age], List[int]], product_rewards,
                              t):
        def _calculate_best_product_rewards(selected_index_, customers_):
            best_product_rewards_ = [0 for _ in self._products]
            criterias = []
            for product in self._products:
                criterias.append(self._select_price_criteria(product))

            for customer_ in customers_:
                _selected_p_index = selected_index_[(customer_.expertise, customer_.age)]
                for product in self._products:
                    clicks = len(customer.products_clicked)
                    if clicks == 0:
                        continue
                    buys = customer_.products_bought[product.id][0]  # Is this count or just 1?
                    ratio = buys / clicks  # - np.sqrt(np.log(self._t) / clicks)  # From the guys code
                    ratio = max([0, ratio])  # But why?
                    estimated_criteria = criterias[product.id][(customer_.expertise, customer_.age)][_selected_p_index[product.id]]
                    estimated_reward = ratio * estimated_criteria
                    if np.isnan(estimated_reward) or np.isinf(estimated_reward):
                        estimated_reward = 0
                    best_product_rewards_[product.id] += estimated_reward

            return best_product_rewards_

        customers = self._experiment_history[-1].customers
        classes = {k: [] for k in selected_price_indexes.keys()}
        for customer in customers:
            classes[(customer.expertise, customer.age)].append(customer)
        ran_for = set()

        for exp in list(Experience):
            for age in list(Age):
                if (exp, age) in self._learners:
                    self._select_learner(exp, age)._update_learner_state(
                        selected_price_indexes[(exp, age)],
                        _calculate_best_product_rewards(selected_price_indexes, classes[(exp, age)]),
                        t)
                    ran_for.add((exp, age))
                elif (exp, None) in self._learners and not (exp,) in ran_for:
                    assert selected_price_indexes[(exp, Age.OLD)] == selected_price_indexes[(exp, Age.YOUNG)]
                    self._select_learner(exp, age)._update_learner_state(
                        selected_price_indexes[(exp, Age.OLD)],
                        # If there is one learner, there can't be two different selected prices so whichever
                        _calculate_best_product_rewards(selected_price_indexes,
                                                        classes[(exp, Age.OLD)] + classes[(exp, Age.YOUNG)]),
                        t)
                    ran_for.add((exp,))
                elif (None, age) in self._learners and not (age,) in ran_for:
                    assert selected_price_indexes[(Experience.PROFESSIONAL, age)] == selected_price_indexes[
                        (Experience.BEGINNER, age)]
                    self._select_learner(exp, age)._update_learner_state(
                        selected_price_indexes[(Experience.PROFESSIONAL, age)],
                        # If there is one learner, there can't be two different selected prices so both are same
                        _calculate_best_product_rewards(selected_price_indexes,
                                                        classes[(Experience.PROFESSIONAL, age)] + classes[
                                                            (Experience.BEGINNER, age)]),
                        t)
                    ran_for.add((age,))
        if not ran_for:
            assert selected_price_indexes[(Experience.PROFESSIONAL, Age.OLD)] == selected_price_indexes[
                (Experience.BEGINNER, Age.OLD)]
            assert selected_price_indexes[(Experience.PROFESSIONAL, Age.OLD)] == selected_price_indexes[
                (Experience.PROFESSIONAL, Age.YOUNG)]
            assert selected_price_indexes[(Experience.PROFESSIONAL, Age.YOUNG)] == selected_price_indexes[
                (Experience.BEGINNER, Age.YOUNG)]
            # There is only the default learner, no splitting yet
            self._learners[(None, None)]._update_learner_state(
                selected_price_indexes[(Experience.PROFESSIONAL, Age.OLD)],
                _calculate_best_product_rewards(selected_price_indexes, customers), t
            )
        if t % 14 != 0:
            return

        bi_weekly_experiment_history = self._experiment_history[-14:]
        bi_weekly_customer_history = [exp.customers for exp in bi_weekly_experiment_history]
        historical_customers = {k: [[] for _ in range(14)] for k in selected_price_indexes.keys()}
        for i, day in enumerate(bi_weekly_customer_history):
            for customer in day:
                historical_customers[(customer.expertise, customer.age)][i].append(customer)

        def split_criteria(subset1, subset2) -> float:
            reward1 = _calculate_best_product_rewards(selected_price_indexes, subset1)
            reward2 = _calculate_best_product_rewards(selected_price_indexes, subset2)
            ratio1 = len(subset1) / len(customers)
            # ratio1 -= 1.65 * np.sqrt(ratio1 * (1 - ratio1) / len(customers))
            ratio2 = len(subset2) / len(customers)
            # ratio2 -= 1.65 * np.sqrt(ratio2 * (1 - ratio2) / len(customers))
            return sum(reward1) * ratio1 + sum(reward2) * ratio2

        def merge(a1, a2):
            return [a1[i] + a2[i] for i in range(14)]

        def split_learners(subset1, subset2):
            learner1 = self._learner_factory(self.config)
            learner2 = self._learner_factory(self.config)
            for i in range(14):
                price_indexes = bi_weekly_experiment_history[i][1]

                learner1._update_learner_state(price_indexes, _calculate_best_product_rewards(price_indexes,
                                                                                              subset1[i]), i + 1)
                learner2._update_learner_state(price_indexes, _calculate_best_product_rewards(price_indexes,
                                                                                              subset2[i]), i + 1)
            return learner1, learner2

        experience_split_beginner = classes[(Experience.BEGINNER, Age.YOUNG)] + classes[(Experience.BEGINNER, Age.OLD)]
        experience_split_pros = classes[(Experience.PROFESSIONAL, Age.YOUNG)] + classes[
            (Experience.PROFESSIONAL, Age.OLD)]

        split_experience_criteria = split_criteria(experience_split_beginner, experience_split_pros)
        age_split_young = classes[(Experience.BEGINNER, Age.YOUNG)] + classes[(Experience.PROFESSIONAL, Age.YOUNG)]
        age_split_old = classes[(Experience.BEGINNER, Age.OLD)] + classes[(Experience.PROFESSIONAL, Age.OLD)]

        split_age_criteria = split_criteria(age_split_young, age_split_old)
        base_criteria = sum(_calculate_best_product_rewards(selected_price_indexes, customers))
        if split_experience_criteria > base_criteria and (
                split_age_criteria <= base_criteria or split_experience_criteria > split_age_criteria):
            beginner_learner, professional_learner = split_learners(
                merge(historical_customers[(Experience.BEGINNER, Age.YOUNG)],
                      historical_customers[(Experience.BEGINNER, Age.OLD)]),
                merge(historical_customers[(Experience.PROFESSIONAL, Age.YOUNG)],
                      historical_customers[(Experience.PROFESSIONAL, Age.OLD)]))
            split_age_beginner = split_criteria(classes[(Experience.BEGINNER, Age.YOUNG)],
                                                classes[(Experience.BEGINNER, Age.OLD)]) > sum(
                _calculate_best_product_rewards(selected_price_indexes, experience_split_beginner))
            split_age_pros = split_criteria(classes[(Experience.PROFESSIONAL, Age.YOUNG)],
                                            classes[(Experience.PROFESSIONAL, Age.OLD)]) > base_criteria > sum(
                _calculate_best_product_rewards(selected_price_indexes, experience_split_pros))
            if split_age_beginner:
                self._learners[(Experience.BEGINNER, Age.YOUNG)], self._learners[
                    (Experience.BEGINNER, Age.OLD)] = split_learners(
                    historical_customers[(Experience.BEGINNER, Age.YOUNG)],
                    historical_customers[(Experience.BEGINNER, Age.OLD)])
            else:
                self._learners[(Experience.BEGINNER, None)] = beginner_learner
            if split_age_pros:
                self._learners[(Experience.PROFESSIONAL, Age.YOUNG)], self._learners[
                    (Experience.PROFESSIONAL, Age.OLD)] = split_learners(
                    historical_customers[(Experience.PROFESSIONAL, Age.YOUNG)],
                    historical_customers[(Experience.PROFESSIONAL, Age.OLD)])
            else:
                self._learners[(Experience.PROFESSIONAL, None)] = professional_learner
        elif split_age_criteria > base_criteria:
            young_learner, old_learner = split_learners(merge(historical_customers[(Experience.BEGINNER, Age.YOUNG)],
                                                              historical_customers[
                                                                  (Experience.PROFESSIONAL, Age.YOUNG)]),
                                                        merge(historical_customers[(Experience.BEGINNER, Age.OLD)],
                                                              historical_customers[(Experience.PROFESSIONAL, Age.OLD)]))
            split_experience_young = split_criteria(classes[(Experience.BEGINNER, Age.YOUNG)],
                                                    classes[(Experience.PROFESSIONAL, Age.YOUNG)]) > sum(
                _calculate_best_product_rewards(selected_price_indexes, age_split_young))
            split_experience_old = split_criteria(classes[(Experience.BEGINNER, Age.OLD)],
                                                  classes[(Experience.PROFESSIONAL, Age.OLD)]) > base_criteria > sum(
                _calculate_best_product_rewards(selected_price_indexes, age_split_old))
            if split_experience_young:
                self._learners[(Experience.BEGINNER, Age.YOUNG)], self._learners[
                    (Experience.PROFESSIONAL, Age.YOUNG)] = split_learners(
                    historical_customers[(Experience.BEGINNER, Age.YOUNG)],
                    historical_customers[(Experience.PROFESSIONAL, Age.YOUNG)])
            else:
                self._learners[(None, Age.YOUNG)] = young_learner
            if split_experience_old:
                self._learners[(Experience.BEGINNER, Age.OLD)], self._learners[
                    (Experience.PROFESSIONAL, Age.OLD)] = split_learners(
                    historical_customers[(Experience.BEGINNER, Age.OLD)],
                    historical_customers[(Experience.PROFESSIONAL, Age.OLD)])
            else:
                self._learners[(None, Age.OLD)] = old_learner
