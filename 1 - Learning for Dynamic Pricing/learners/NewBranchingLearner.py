import enum
from typing import List, Union, Dict, Tuple, NamedTuple, Optional

import numpy as np

from BanditLearner import BanditLearner
from basic_types import Experience, Age
from entities import Product
from itertools import product as prod


class NewBranchingLearner(BanditLearner):
    def _upper_bound(self):
        return 0

    def __init__(self, config, learner_factory):
        super().__init__(config)
        self.config = config
        self._learner_factory = learner_factory
        self._reset_parameters()


    def _reset_parameters(self):
        def _make():
            new_learner = self._learner_factory(self.config._replace(context_generation=False))
            if hasattr(self, "_vars"):
                new_learner.refresh_vars(*self._vars)
                new_learner._reset_parameters()
            return new_learner

        self._active_learners = set()
        self._learners: Dict[Tuple[Optional[Experience], Optional[Age]], BanditLearner] = {
            (None, None): _make()}
        for exp in list(Experience):
            for age in list(Age):
                self._learners[(exp, age)] = _make()
        for exp in list(Experience):
            self._learners[(exp, None)] = _make()
        for age in list(Age):
            self._learners[(None, age)] = _make()

    def refresh_vars(self, _products, _environment, _config):
        super(NewBranchingLearner, self).refresh_vars(_products, _environment, _config)
        self._vars = (_products, _environment, _config)
        self._environment = _environment
        self._config = _config
        self._products = _products
        self._active_learners = set()
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
                if (exp, age) in self._active_learners:
                    x[(exp, age)] = self._select_learner(exp, age)._select_price_criteria(product)
                    ran_for.add((exp, age))
                elif (exp, None) in self._active_learners and exp not in ran_for:
                    x[(exp, age)] = self._select_learner(exp, age)._select_price_criteria(product)
                    x[(exp, [age_ for age_ in list(Age) if age_ != age][0])] = x[(exp, age)]
                    ran_for.add(exp)
                elif (None, age) in self._active_learners and age not in ran_for:
                    x[(exp, age)] = self._select_learner(exp, age)._select_price_criteria(product)
                    x[([exp_ for exp_ in list(Experience) if exp_ != exp][0], age)] = x[(exp, age)]
                    ran_for.add(age)

        if not ran_for:
            # There is only the default learner, no splitting yet
            val = self._learners[(None, None)]._select_price_criteria(product)
            x = {(e, a): val for e, a in prod(list(Experience), list(Age))}
        return x

    def _update_learner_state(self, selected_price_indexes: Dict[Tuple[Experience, Age], List[int]], product_rewards,
                              t):
        customers = self._experiment_history[-1].customers
        classes = {k: [] for k in selected_price_indexes.keys()}
        for customer in customers:
            classes[(customer.expertise, customer.age)].append(customer)

        def _calculate_product_rewards(selected_price_indexes, customers: List["Customer"]):
            product_rewards = self._calculate_product_rewards(selected_price_indexes, customers)
            if self._t > 1:
                for estimator in self._estimators:
                    product_rewards = estimator.modify(product_rewards)
            return product_rewards

        for exp in list(Experience):
            for age in list(Age):
                p_indexes = selected_price_indexes[(exp, age)]
                s_customers = classes[(exp, age)]
                self._learners[(exp, age)]._update_learner_state(p_indexes,
                                                                 _calculate_product_rewards(p_indexes,
                                                                                                 s_customers),
                                                                 t
                                                                 )
        for exp in list(Experience):
            p_indexes = selected_price_indexes[(exp, Age.OLD)]
            s_customers = classes[(exp, Age.OLD)] + classes[(exp, Age.YOUNG)]
            self._learners[(exp, None)]._update_learner_state(p_indexes,
                                                              _calculate_product_rewards(p_indexes, s_customers),
                                                              t
                                                              )
        for age in list(Age):
            p_indexes = selected_price_indexes[(Experience.BEGINNER, age)]
            s_customers = classes[(Experience.BEGINNER, age)] + classes[(Experience.PROFESSIONAL, age)]
            self._learners[(None, age)]._update_learner_state(p_indexes,
                                                              _calculate_product_rewards(p_indexes, s_customers),
                                                              t
                                                              )
        p_indexes = selected_price_indexes[(Experience.PROFESSIONAL, Age.OLD)]
        self._learners[(None, None)]._update_learner_state(
            p_indexes,
            _calculate_product_rewards(p_indexes, customers), t)

        if t % 30 != 0:
            return
        print()
        print("Entered detection, already running are", self._active_learners)
        print()
        bi_weekly_experiment_history = self._experiment_history[-14:]
        bi_weekly_customer_history = [exp.customers for exp in bi_weekly_experiment_history]
        historical_customers = {k: [] for k in selected_price_indexes.keys()}
        for daily_customers in bi_weekly_customer_history:
            for customer in daily_customers:
                historical_customers[(customer.expertise, customer.age)].append(customer)

        def get_base(tpl, c):
            base_indexes = self._learners[tpl]._select_price_indexes()
            return sum(_calculate_product_rewards(base_indexes, c))

        old_customers = historical_customers[(Experience.PROFESSIONAL, Age.OLD)] + historical_customers[
            (Experience.BEGINNER, Age.OLD)]
        young_customers = historical_customers[(Experience.PROFESSIONAL, Age.YOUNG)] + historical_customers[
            (Experience.BEGINNER, Age.YOUNG)]
        professional_customers = historical_customers[(Experience.PROFESSIONAL, Age.YOUNG)] + historical_customers[
            (Experience.PROFESSIONAL, Age.OLD)]
        beginner_customers = historical_customers[(Experience.BEGINNER, Age.YOUNG)] + historical_customers[
            (Experience.BEGINNER, Age.OLD)]

        def split_criteria(lr1: BanditLearner, c1: List["Customer"], lr2: BanditLearner, c2: List["Customer"]) -> float:
            ind1 = lr1._select_price_indexes()
            r1 = _calculate_product_rewards(ind1, c1)
            ind2 = lr2._select_price_indexes()
            r2 = _calculate_product_rewards(ind2, c2)

            ratio1 = len(c1) / len(customers)
            # ratio1 -= 1.65 * np.sqrt(ratio1 * (1 - ratio1) / len(customers))
            ratio2 = len(c2) / len(
                customers)  # Len customers make more sense here since it is the percentage effect of all the cases, but we need to handle the cases with some contribution from base
            # ratio2 -= 1.65 * np.sqrt(ratio2 * (1 - ratio2) / len(customers))
            return sum(r1) * ratio1 + sum(r2) * ratio2

        if (None, Age.OLD) not in self._active_learners and (
                Experience.PROFESSIONAL, None) not in self._active_learners:
            # We can split from age or experience
            split_age = split_criteria(self._learners[(None, Age.OLD)], old_customers,
                                       self._learners[(None, Age.YOUNG)], young_customers)
            split_experience = split_criteria(self._learners[(Experience.PROFESSIONAL, None)], professional_customers,
                                              self._learners[(Experience.BEGINNER, None)], beginner_customers) * 1000
            base_reward = get_base((None, None), customers)
            if split_age > base_reward or split_experience > base_reward:
                if split_age > split_experience:
                    self._active_learners.add((None, Age.YOUNG))
                    self._active_learners.add((None, Age.OLD))
                    self._learners[(None, Age.OLD)]._reset_for_current_time(self._t)
                    self._learners[(None, Age.YOUNG)]._reset_for_current_time(self._t)
                else:
                    self._active_learners.add((Experience.PROFESSIONAL, None))
                    self._active_learners.add((Experience.BEGINNER, None))
                    self._learners[(Experience.PROFESSIONAL, None)]._reset_for_current_time(self._t)
                    self._learners[(Experience.BEGINNER, None)]._reset_for_current_time(self._t)
        elif (None, Age.OLD) not in self._active_learners:
            # We already split from experience
            # We can either split age from professional or from beginner
            base_pro = get_base((Experience.PROFESSIONAL, None), professional_customers) * (
                    len(professional_customers) / len(customers))
            base_beg = get_base((Experience.BEGINNER, None), beginner_customers) * (
                    len(beginner_customers) / len(customers))
            if (Experience.PROFESSIONAL, Age.OLD) not in self._active_learners and (
                    Experience.BEGINNER, Age.OLD) not in self._active_learners:
                # We split from experience, but no split by age is made
                split_age_professional = split_criteria(
                    self._learners[(Experience.PROFESSIONAL, Age.OLD)],
                    classes[(Experience.PROFESSIONAL, Age.OLD)],
                    self._learners[(Experience.PROFESSIONAL, Age.YOUNG)],
                    classes[(Experience.PROFESSIONAL, Age.YOUNG)],
                )
                split_age_beginner = split_criteria(
                    self._learners[(Experience.BEGINNER, Age.OLD)],
                    classes[(Experience.BEGINNER, Age.OLD)],
                    self._learners[(Experience.BEGINNER, Age.YOUNG)],
                    classes[(Experience.BEGINNER, Age.YOUNG)],
                ) * 1000
                if split_age_professional > base_pro and (
                        split_age_beginner < base_beg or split_age_professional - base_pro > split_age_beginner - base_beg):
                    self._active_learners.add((Experience.PROFESSIONAL, Age.YOUNG))
                    self._active_learners.add((Experience.PROFESSIONAL, Age.OLD))
                    self._learners[(Experience.PROFESSIONAL, Age.OLD)]._reset_for_current_time(self._t)
                    self._learners[(Experience.PROFESSIONAL, Age.YOUNG)]._reset_for_current_time(self._t)
                elif split_age_beginner > base_beg:
                    self._active_learners.add((Experience.BEGINNER, Age.YOUNG))
                    self._active_learners.add((Experience.BEGINNER, Age.OLD))
                    self._learners[(Experience.BEGINNER, Age.OLD)]._reset_for_current_time(self._t)
                    self._learners[(Experience.BEGINNER, Age.YOUNG)]._reset_for_current_time(self._t)

            elif (Experience.PROFESSIONAL, Age.OLD) not in self._active_learners:
                # We split age for beginner, not yet professional
                split_age_professional = split_criteria(
                    self._learners[(Experience.PROFESSIONAL, Age.OLD)],
                    classes[(Experience.PROFESSIONAL, Age.OLD)],
                    self._learners[(Experience.PROFESSIONAL, Age.YOUNG)],
                    classes[(Experience.PROFESSIONAL, Age.YOUNG)],
                )
                if split_age_professional > base_pro:
                    self._active_learners.add((Experience.PROFESSIONAL, Age.YOUNG))
                    self._active_learners.add((Experience.PROFESSIONAL, Age.OLD))
                    return
            elif (Experience.BEGINNER, Age.OLD) not in self._active_learners:
                # We split age for professional, not yet beginner
                split_age_beginner = split_criteria(
                    self._learners[(Experience.BEGINNER, Age.OLD)],
                    classes[(Experience.BEGINNER, Age.OLD)],
                    self._learners[(Experience.BEGINNER, Age.YOUNG)],
                    classes[(Experience.BEGINNER, Age.YOUNG)],
                )
                if split_age_beginner > base_beg:
                    self._active_learners.add((Experience.BEGINNER, Age.YOUNG))
                    self._active_learners.add((Experience.BEGINNER, Age.OLD))
                    self._learners[(Experience.BEGINNER, Age.OLD)]._reset_for_current_time(self._t)
                    self._learners[(Experience.BEGINNER, Age.YOUNG)]._reset_for_current_time(self._t)

            else:
                pass  # Done it all
        elif (Experience.PROFESSIONAL, None) not in self._active_learners:
            # Already split from age
            # Same as above

            base_young = get_base((None, Age.YOUNG), young_customers) * (len(young_customers) / len(customers))
            base_old = get_base((None, Age.OLD), old_customers) * (len(old_customers) / len(customers))

            if (Experience.PROFESSIONAL, Age.OLD) not in self._active_learners and (
                    Experience.PROFESSIONAL, Age.YOUNG) not in self._active_learners:
                # We split from age but not yet experience
                split_experience_young = split_criteria(
                    self._learners[(Experience.PROFESSIONAL, Age.YOUNG)],
                    classes[(Experience.PROFESSIONAL, Age.YOUNG)],
                    self._learners[(Experience.BEGINNER, Age.YOUNG)],
                    classes[(Experience.BEGINNER, Age.YOUNG)],
                )
                split_experience_old = split_criteria(
                    self._learners[(Experience.PROFESSIONAL, Age.OLD)],
                    classes[(Experience.PROFESSIONAL, Age.OLD)],
                    self._learners[(Experience.BEGINNER, Age.OLD)],
                    classes[(Experience.BEGINNER, Age.OLD)]
                )
                if split_experience_old > base_old and (
                        split_experience_young < base_young or split_experience_old - base_old > split_experience_young - base_young):
                    # Split experience from old branch
                    self._active_learners.add((Experience.PROFESSIONAL, Age.OLD))
                    self._active_learners.add((Experience.BEGINNER, Age.OLD))
                    self._learners[(Experience.PROFESSIONAL, Age.OLD)]._reset_for_current_time(self._t)
                    self._learners[(Experience.BEGINNER, Age.OLD)]._reset_for_current_time(self._t)
                elif split_experience_young > base_young:
                    # Split experience from young branch
                    self._active_learners.add((Experience.PROFESSIONAL, Age.YOUNG))
                    self._active_learners.add((Experience.BEGINNER, Age.YOUNG))
                    self._learners[(Experience.PROFESSIONAL, Age.YOUNG)]._reset_for_current_time(self._t)
                    self._learners[(Experience.BEGINNER, Age.YOUNG)]._reset_for_current_time(self._t)

            elif (Experience.PROFESSIONAL, Age.OLD) not in self._active_learners:
                # We split experience from young but not old
                split_experience_old = split_criteria(
                    self._learners[(Experience.PROFESSIONAL, Age.OLD)],
                    classes[(Experience.PROFESSIONAL, Age.OLD)],
                    self._learners[(Experience.BEGINNER, Age.OLD)],
                    classes[(Experience.BEGINNER, Age.OLD)]
                )
                if split_experience_old > base_old:
                    # Split experience from old branch
                    self._active_learners.add((Experience.PROFESSIONAL, Age.OLD))
                    self._active_learners.add((Experience.BEGINNER, Age.OLD))
            elif (Experience.PROFESSIONAL, Age.YOUNG) not in self._active_learners:
                # We already split experience from old branch but not young
                split_experience_young = split_criteria(
                    self._learners[(Experience.PROFESSIONAL, Age.YOUNG)],
                    classes[(Experience.PROFESSIONAL, Age.YOUNG)],
                    self._learners[(Experience.BEGINNER, Age.YOUNG)],
                    classes[(Experience.BEGINNER, Age.YOUNG)],
                )
                if split_experience_young > base_young:
                    # Split experience from young branch
                    self._active_learners.add((Experience.PROFESSIONAL, Age.YOUNG))
                    self._active_learners.add((Experience.BEGINNER, Age.YOUNG))
                    self._learners[(Experience.PROFESSIONAL, Age.YOUNG)]._reset_for_current_time(self._t)
                    self._learners[(Experience.BEGINNER, Age.YOUNG)]._reset_for_current_time(self._t)
            else:
                pass  # Done
        else:
            raise RuntimeError("WTF?")  # Both of them can't be the base split condition

