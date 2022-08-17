from typing import NamedTuple, List, Optional, Dict

from Learner import Reward, PriceIndexes, ProductRewards, Learner
from change_detectors import ChangeHistoryItem
from entities import Product, Customer, Simulation
from parameter_estimators import HistoryEntry


class IDs(NamedTuple):
    storage: str = "storage"
    left_experiment_selector: str = "experiment-selector-left"
    right_experiment_selector: str = "experiment-selector-right"
    result_div_left: str = "result-div-left"
    result_div_right: str = "result-div-right"
    run_experiment: str = "run-experiment"
    reset_results: str = "reset-results"
    run_count: str = "run-count"


ids = IDs()


class SimulationResult(NamedTuple):
    rewards: List[Reward]
    price_indexes: List[PriceIndexes]
    product_rewards: List[ProductRewards]
    products: List[Product]
    customers: Optional[List[List[Customer]]]
    estimators: Optional[Dict[str, List[HistoryEntry]]]
    change_detected_at: List[int]
    change_history: Optional[List[ChangeHistoryItem]]

    def serialize(self):
        result = {
            "rewards": self.rewards,
            "price_indexes": self.price_indexes,
            "product_rewards": self.product_rewards,
            "products": [product.serialize() for product in self.products],
            "change_detected_at": self.change_detected_at,
            "change_history": self.change_history,
        }
        if self.customers is not None:
            result["customers"] = [[customer.serialize() for customer in day] for day in self.customers]
        if self.estimators is not None:
            result["estimators"] = self.estimators
        return result

    @staticmethod
    def deserialize(data):
        customers = data.get("customers")
        if customers is not None:
            customers = [[Customer(*customer.values()) for customer in day] for day in customers]
        return SimulationResult(
            rewards=data["rewards"],
            price_indexes=data["price_indexes"],
            product_rewards=data["product_rewards"],
            products=[Product(*product.values()) for product in data["products"]],
            customers=customers,
            estimators=data.get("estimators"),
            change_detected_at=data["change_detected_at"],
            change_history=data["change_history"]
        )

    @staticmethod
    def from_result(learner: Learner, simulation: Simulation):
        exps = learner._experiment_history
        rewards = [reward for (reward, _, _, _, _) in exps]
        price_indexes = [price_indexes for (_, price_indexes, _, _, _) in exps]
        product_rewards = [product_rewards for (_, _, product_rewards, _, _) in exps]
        customer_history = None
        if hasattr(learner, "_customer_history"):
            customer_history = learner._customer_history
        estimators = None
        if hasattr(learner, "_estimators"):
            estimators = {type(estimator).__name__: estimator._history for estimator in learner._estimators}
        change_indexes = [ind for ind, value in enumerate(exps) if value[3]]

        if exps[0][4] is not None:
            change_history = [change for (_, _, _, _, change) in exps]
        else:
            change_history = []
        return SimulationResult(rewards, price_indexes, product_rewards, simulation.products, customer_history,
                                estimators, change_detected_at=change_indexes, change_history=change_history)
